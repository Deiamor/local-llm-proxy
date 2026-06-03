#!/usr/bin/env bun
/**
 * Anthropic Messages API → OpenAI Chat Completions proxy
 *
 * Bridges claude-code-main (Anthropic SDK) to a local llama.cpp server.
 *
 * Usage:
 *   bun supergemma-proxy.ts
 *
 * Then set in .env:
 *   ANTHROPIC_BASE_URL=http://127.0.0.1:4000
 *   ANTHROPIC_API_KEY=fake-key
 *   ANTHROPIC_MODEL=supergemma4-26b-Q4_K_M.gguf
 */

import http from 'node:http'

const UPSTREAM = process.env.LLAMA_URL ?? process.env.SUPERGEMMA_URL ?? 'http://127.0.0.1:8080'
const PORT = Number(process.env.PROXY_PORT ?? 4000)
const DEFAULT_MODEL = process.env.ANTHROPIC_MODEL ?? 'local-model'

// ── Type definitions ────────────────────────────────────────────────────────

interface TextBlock { type: 'text'; text: string }
interface ToolUseBlock { type: 'tool_use'; id: string; name: string; input: unknown }
interface ToolResultBlock {
  type: 'tool_result'
  tool_use_id: string
  content: string | TextBlock[]
}
interface ImageBlock { type: 'image'; source: unknown }

type ContentBlock = TextBlock | ToolUseBlock | ToolResultBlock | ImageBlock

interface AnthropicMessage { role: 'user' | 'assistant'; content: string | ContentBlock[] }

interface AnthropicTool {
  name: string
  description?: string
  input_schema: object
}

interface AnthropicRequest {
  model?: string
  messages: AnthropicMessage[]
  system?: string
  max_tokens?: number
  temperature?: number
  top_p?: number
  stream?: boolean
  tools?: AnthropicTool[]
  tool_choice?: unknown
  stop_sequences?: string[]
}

// ── Request conversion: Anthropic → OpenAI ──────────────────────────────────

function toOpenAIMessages(messages: AnthropicMessage[], system?: string): unknown[] {
  const result: unknown[] = []
  if (system) result.push({ role: 'system', content: system })
  for (const msg of messages) {
    const converted = convertMessage(msg)
    if (Array.isArray(converted)) result.push(...converted)
    else result.push(converted)
  }
  return result
}

function convertMessage(msg: AnthropicMessage): unknown | unknown[] {
  if (typeof msg.content === 'string') {
    return { role: msg.role, content: msg.content }
  }

  const textParts: string[] = []
  const toolCalls: unknown[] = []
  const toolResults: unknown[] = []

  for (const block of msg.content) {
    switch (block.type) {
      case 'text':
        textParts.push(block.text)
        break
      case 'tool_use':
        toolCalls.push({
          id: block.id,
          type: 'function',
          function: { name: block.name, arguments: JSON.stringify(block.input ?? {}) }
        })
        break
      case 'tool_result': {
        let content: string
        if (typeof block.content === 'string') {
          content = block.content
        } else if (Array.isArray(block.content)) {
          content = block.content
            .filter((c): c is TextBlock => c.type === 'text')
            .map(c => c.text)
            .join('\n')
        } else {
          content = ''
        }
        toolResults.push({ role: 'tool', tool_call_id: block.tool_use_id, content })
        break
      }
      case 'image':
        textParts.push('[image]')
        break
    }
  }

  if (toolResults.length > 0) return toolResults

  if (toolCalls.length > 0) {
    return {
      role: 'assistant',
      content: textParts.join('\n') || null,
      tool_calls: toolCalls
    }
  }

  return { role: msg.role, content: textParts.join('\n') }
}

function toOpenAITools(tools: AnthropicTool[]): unknown[] {
  return tools.map(t => ({
    type: 'function',
    function: { name: t.name, description: t.description, parameters: t.input_schema }
  }))
}

function toOpenAIToolChoice(toolChoice: unknown): unknown {
  if (!toolChoice || typeof toolChoice !== 'object') return undefined
  const tc = toolChoice as { type: string; name?: string }
  if (tc.type === 'auto') return 'auto'
  if (tc.type === 'any') return 'required'
  if (tc.type === 'tool' && tc.name) return { type: 'function', function: { name: tc.name } }
  return undefined
}

// ── Response conversion: OpenAI → Anthropic (non-streaming) ────────────────

function fromOpenAIResponse(openaiResp: Record<string, unknown>, model: string): unknown {
  const choices = openaiResp.choices as Array<Record<string, unknown>>
  const choice = choices?.[0]
  if (!choice) throw new Error('No choices in upstream response')

  const message = choice.message as Record<string, unknown>
  const content: unknown[] = []

  const rawContent = (message.content || (message as Record<string, unknown>).reasoning_content) as string | null
  if (rawContent) content.push({ type: 'text', text: rawContent })

  const toolCalls = message.tool_calls as Array<Record<string, unknown>> | undefined
  if (toolCalls) {
    for (const tc of toolCalls) {
      const fn = tc.function as Record<string, string>
      let input: unknown
      try { input = JSON.parse(fn.arguments || '{}') } catch { input = {} }
      content.push({ type: 'tool_use', id: tc.id, name: fn.name, input })
    }
  }

  const usage = openaiResp.usage as Record<string, number> | undefined

  return {
    id: openaiResp.id ?? `msg_${Date.now()}`,
    type: 'message',
    role: 'assistant',
    content,
    model,
    stop_reason: mapStopReason(choice.finish_reason as string | null),
    stop_sequence: null,
    usage: {
      input_tokens: usage?.prompt_tokens ?? 0,
      output_tokens: usage?.completion_tokens ?? 0
    }
  }
}

function mapStopReason(r: string | null): string {
  if (r === 'tool_calls') return 'tool_use'
  if (r === 'length') return 'max_tokens'
  return 'end_turn'
}

// ── Streaming conversion: OpenAI SSE → Anthropic SSE ──────────────────────

async function pipeStream(
  upstreamBody: ReadableStream<Uint8Array>,
  model: string,
  res: http.ServerResponse
): Promise<void> {
  const msgId = `msg_${Date.now()}`
  let blockIndex = 0
  let hasOpenBlock = false
  let outputTokens = 0
  let stopReason = 'end_turn'

  // Track tool call accumulation across deltas (keyed by OpenAI tc.index)
  const toolCallBlocks = new Map<number, { blockIdx: number; id: string; name: string }>()

  function sse(event: string, data: unknown) {
    res.write(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`)
  }

  sse('message_start', {
    type: 'message_start',
    message: {
      id: msgId, type: 'message', role: 'assistant',
      content: [], model,
      stop_reason: null, stop_sequence: null,
      usage: { input_tokens: 0, output_tokens: 1 }
    }
  })
  sse('ping', { type: 'ping' })

  const reader = upstreamBody.getReader()
  const dec = new TextDecoder()
  let buf = ''

  try {
    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      buf += dec.decode(value, { stream: true })
      const lines = buf.split('\n')
      buf = lines.pop() ?? ''

      for (const line of lines) {
        if (!line.startsWith('data: ')) continue
        const payload = line.slice(6).trim()
        if (payload === '[DONE]') continue

        let chunk: Record<string, unknown>
        try { chunk = JSON.parse(payload) } catch { continue }

        const choices = chunk.choices as Array<Record<string, unknown>> | undefined
        const choice = choices?.[0]
        if (!choice) continue

        const delta = (choice.delta ?? {}) as Record<string, unknown>

        if (chunk.usage) {
          const u = chunk.usage as Record<string, number>
          outputTokens = u.completion_tokens ?? outputTokens
        }

        if (choice.finish_reason) {
          stopReason = mapStopReason(choice.finish_reason as string)
        }

        // Skip reasoning_content (internal thinking tokens — not forwarded to client)
        // delta.reasoning_content exists on thinking-mode models like supergemma4

        // Text delta
        const textContent = delta.content as string | undefined
        if (textContent) {
          if (!hasOpenBlock) {
            sse('content_block_start', {
              type: 'content_block_start',
              index: blockIndex,
              content_block: { type: 'text', text: '' }
            })
            hasOpenBlock = true
          }
          sse('content_block_delta', {
            type: 'content_block_delta',
            index: blockIndex,
            delta: { type: 'text_delta', text: textContent }
          })
          outputTokens++
        }

        // Tool call deltas
        const toolCallDeltas = delta.tool_calls as Array<Record<string, unknown>> | undefined
        if (toolCallDeltas) {
          for (const tc of toolCallDeltas) {
            const tcIdx = (tc.index as number) ?? 0
            const fn = (tc.function ?? {}) as Record<string, string>

            if (tc.id) {
              // New tool call — close current text block if any
              if (hasOpenBlock) {
                sse('content_block_stop', { type: 'content_block_stop', index: blockIndex })
                blockIndex++
                hasOpenBlock = false
              }
              toolCallBlocks.set(tcIdx, { blockIdx: blockIndex, id: tc.id as string, name: fn.name ?? '' })
              sse('content_block_start', {
                type: 'content_block_start',
                index: blockIndex,
                content_block: { type: 'tool_use', id: tc.id, name: fn.name ?? '', input: {} }
              })
              hasOpenBlock = true
            }

            if (fn.name && !toolCallBlocks.get(tcIdx)?.name) {
              const entry = toolCallBlocks.get(tcIdx)
              if (entry) entry.name = fn.name
            }

            if (fn.arguments) {
              const entry = toolCallBlocks.get(tcIdx)
              const idx = entry?.blockIdx ?? blockIndex
              sse('content_block_delta', {
                type: 'content_block_delta',
                index: idx,
                delta: { type: 'input_json_delta', partial_json: fn.arguments }
              })
            }
          }
        }
      }
    }
  } finally {
    reader.releaseLock()
  }

  if (hasOpenBlock) {
    sse('content_block_stop', { type: 'content_block_stop', index: blockIndex })
  }

  console.error(`[proxy] ← stopReason=${stopReason} outputTokens=${outputTokens} blocks=${blockIndex + (hasOpenBlock ? 0 : 0)}`)
  sse('message_delta', {
    type: 'message_delta',
    delta: { stop_reason: stopReason, stop_sequence: null },
    usage: { output_tokens: outputTokens || 1 }
  })
  sse('message_stop', { type: 'message_stop' })
}

// ── HTTP Server ─────────────────────────────────────────────────────────────

const server = http.createServer(async (req, res) => {
  console.error(`[proxy] ${req.method} ${req.url}`)
  const headers: Record<string, string> = { 'Access-Control-Allow-Origin': '*' }

  if (req.method === 'OPTIONS') {
    res.writeHead(204, {
      ...headers,
      'Access-Control-Allow-Methods': 'POST, GET, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type, Authorization, x-api-key, anthropic-version, anthropic-beta'
    })
    res.end()
    return
  }

  // Health / models endpoint
  if (req.url === '/v1/models' || req.url === '/') {
    res.writeHead(200, { ...headers, 'Content-Type': 'application/json' })
    res.end(JSON.stringify({ data: [{ id: DEFAULT_MODEL, object: 'model' }] }))
    return
  }

  const reqPath = req.url?.split('?')[0] ?? ''
  if (reqPath !== '/v1/messages' || req.method !== 'POST') {
    res.writeHead(404, { ...headers, 'Content-Type': 'application/json' })
    res.end(JSON.stringify({ error: { message: 'Not found', type: 'not_found_error' } }))
    return
  }

  // Collect request body
  const chunks: Buffer[] = []
  for await (const chunk of req) chunks.push(chunk as Buffer)
  const body = Buffer.concat(chunks).toString()

  let anthropicReq: AnthropicRequest
  try {
    anthropicReq = JSON.parse(body) as AnthropicRequest
    // Debug: log last user message
    const msgs = anthropicReq.messages ?? []
    const last = msgs[msgs.length - 1]
    const preview = typeof last?.content === 'string' ? last.content.slice(0, 80) : JSON.stringify(last?.content).slice(0, 80)
    console.error(`[proxy] → role=${last?.role} content=${preview}`)
  } catch {
    res.writeHead(400, { ...headers, 'Content-Type': 'application/json' })
    res.end(JSON.stringify({ error: { message: 'Invalid JSON', type: 'invalid_request_error' } }))
    return
  }

  // Pass max_tokens directly — server runs with --reasoning off so no thinking tokens
  const requestedMaxTokens = anthropicReq.max_tokens ?? 8192
  const upstreamMaxTokens = requestedMaxTokens

  const openaiBody: Record<string, unknown> = {
    model: DEFAULT_MODEL,
    messages: toOpenAIMessages(anthropicReq.messages, anthropicReq.system),
    max_tokens: upstreamMaxTokens,
    stream: anthropicReq.stream ?? false,
    // Low temperature for reliable tool calling
    temperature: anthropicReq.temperature ?? 0.1,
  }
  if (anthropicReq.temperature != null) openaiBody.temperature = anthropicReq.temperature
  if (anthropicReq.top_p != null) openaiBody.top_p = anthropicReq.top_p
  if (anthropicReq.stop_sequences?.length) openaiBody.stop = anthropicReq.stop_sequences
  if (anthropicReq.tools?.length) {
    openaiBody.tools = toOpenAITools(anthropicReq.tools)
    const tc = toOpenAIToolChoice(anthropicReq.tool_choice)
    // Default to 'auto' so model can freely choose tool or text
    openaiBody.tool_choice = tc ?? 'auto'
  }

  let upstreamRes: Response
  try {
    upstreamRes = await fetch(`${UPSTREAM}/v1/chat/completions`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(openaiBody)
    })
  } catch (err) {
    res.writeHead(502, { ...headers, 'Content-Type': 'application/json' })
    res.end(JSON.stringify({ error: { message: `Upstream unreachable: ${err}`, type: 'api_error' } }))
    return
  }

  if (!upstreamRes.ok) {
    const errText = await upstreamRes.text()
    res.writeHead(upstreamRes.status, { ...headers, 'Content-Type': 'application/json' })
    res.end(JSON.stringify({ error: { message: errText, type: 'api_error' } }))
    return
  }

  const requestedModel = anthropicReq.model ?? DEFAULT_MODEL

  if (anthropicReq.stream) {
    res.writeHead(200, {
      ...headers,
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-cache',
      'Connection': 'keep-alive'
    })
    await pipeStream(upstreamRes.body!, requestedModel, res)
    res.end()
  } else {
    const openaiResp = await upstreamRes.json() as Record<string, unknown>
    let anthropicResp: unknown
    try {
      anthropicResp = fromOpenAIResponse(openaiResp, requestedModel)
    } catch (err) {
      res.writeHead(500, { ...headers, 'Content-Type': 'application/json' })
      res.end(JSON.stringify({ error: { message: String(err), type: 'api_error' } }))
      return
    }
    res.writeHead(200, { ...headers, 'Content-Type': 'application/json' })
    res.end(JSON.stringify(anthropicResp))
  }
})

server.listen(PORT, '127.0.0.1', () => {
  console.log(`[proxy] Anthropic → OpenAI proxy running on http://127.0.0.1:${PORT}`)
  console.log(`[proxy] Upstream: ${UPSTREAM}`)
  console.log(`[proxy] Model: ${DEFAULT_MODEL}`)
})
