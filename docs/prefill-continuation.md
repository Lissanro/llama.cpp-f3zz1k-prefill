# Prefill / continuation support in llama.cpp

This document describes how to resume ("prefill") an interrupted assistant
message with `llama-server`, including reasoning, content, and tool calls.

## Table of contents

- [What prefill is](#what-prefill-is)
- [API fields](#api-fields)
- [How it works internally](#how-it-works-internally)
- [Prefill cases](#prefill-cases)
- [Tool-call prefill and `__raw`](#tool-call-prefill-and-__raw)
- [Streaming behavior](#streaming-behavior)
- [Test commands](#test-commands)
- [Limitations and notes](#limitations-and-notes)

## What prefill is

In a normal chat completion request the server appends the assistant prefix
(e.g. `<|im_start|>assistant\n`) to the prompt and lets the model generate the
rest of the assistant turn.

Prefill lets the client provide the first tokens of that assistant turn. The
server places those tokens into the model's KV cache and continues generation
from there. This is useful for:

- Resuming an assistant message that was interrupted (e.g. network error,
  max_tokens hit, client cancellation).
- Steering the assistant by fixing the start of its reply.
- Providing in-context tool-call examples.

Prefill is requested by putting an `assistant` message at the **end** of the
`messages` array and enabling the server's prefill option. With the OpenAI
compatible endpoint this happens automatically when the last message has role
`assistant`.

## API fields

### Request fields (trailing assistant message)

| Field | Type | Description |
|-------|------|-------------|
| `reasoning_content` | string | Prefilled reasoning/thinking text. |
| `content` | string / null / absent | Prefilled content text. `null` or absent keeps a reasoning block open; `""` closes it. |
| `tool_calls` | array | **Complete** structured tool calls (model-agnostic). Allowed only for complete calls. |
| `tool_calls_raw` | string | Exact raw model-format tool-call tokens. Use this for **partial** tool calls; also accepted for complete calls. |

Only one style of tool-call prefill should be used at a time:

- `tool_calls` — complete structured calls. The server serializes them to the
  model's raw format internally.
- `tool_calls_raw` — raw tokens. Required when the call is partial; also works
  for complete calls.

If `tool_calls_raw` (or `tool_calls`) is present, the assistant's `content`
field is considered closed and any model generation happens in the tool-call
section.

### Request option

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `return_prefill` | bool | `false` | Whether the response should include the prefilled text/calls or only the newly generated part. |

### Response fields

| Field | Location | Description |
|-------|----------|-------------|
| `message.reasoning_content` | `choices[0].message` | Reasoning text (prefill + generated when `return_prefill=true`). |
| `message.content` | `choices[0].message` | Content text (prefill + generated when `return_prefill=true`). |
| `message.tool_calls` | `choices[0].message` | Structured tool calls. |
| `tool_calls[i].__raw` | each tool call object | Exact raw model-format tokens for this call. Non-streaming: full call. Streaming: delta of raw tokens. |

The `__raw` field is the key to continuation: a client that was interrupted
can save the cumulative `__raw` string and send it back as `tool_calls_raw`.

## How it works internally

### Prompt construction

When the last message has role `assistant`, the server:

1. Removes that message from the list used for the prompt template.
2. Builds the prompt as if the assistant turn is incomplete.
3. Tokenizes the prefilled text separately and appends it after the assistant
   prefix in the slot's token sequence.

### Parser continuation

The server does **not** manually split generated text for prefill cases. Instead
it sets the PEG parser's `generation_prompt` to the prefilled text. The parser
then sees:

```
generation_prompt + generated_text
= prefill_text + generated_text
```

This lets the same parser extract reasoning, content, and tool calls from the
combined string, exactly as it does for a non-prefilled response.

`is_continuation` is set to `true` so the initial parser state contains the
prefill. This makes streaming diffs naturally exclude the prefill (the client
only receives newly generated tokens). For `return_prefill=true` the server
emits the prefill as synthetic first deltas.

### Tool-call raw tokens

Tool calls are internally represented as structured objects, but models emit
raw tokens (`<tool_call>`, `<function=...>`, etc.). To make continuation
possible the PEG mapper now captures the exact span of raw tokens that produced
each structured tool call and stores it in the `raw` field.

For non-streaming responses `tool_calls[i].__raw` is the full raw text of that
call, including any section/call markers. For streaming responses `__raw` is a
**delta**: each chunk contains only the raw tokens that appeared in that chunk.
Concatenating all `__raw` deltas for an index reproduces the full raw call.

### Partial tool-call prefill

If `tool_calls_raw` does **not** end at a recognized call boundary
(`per_call_end`, `section_end`, or valid JSON array), the server treats it as
partial and disables the tool-call grammar. The grammar is built for generated
text that starts fresh; with a partial prefill it would force the model to
regenerate the `<tool_call>` prefix. Disabling it lets the model continue the
raw tokens, and the PEG parser extracts the completed call afterwards.

For partial prefill the completed tool call is kept in the response even when
`return_prefill=false`, because the client needs the final structured result.
The client can compute the model's raw contribution by subtracting its own
prefill from `__raw`.

## Prefill cases

The server detects the prefill case from the trailing assistant message:

| Case | `reasoning_content` | `content` | Behavior |
|------|---------------------|-----------|----------|
| 1 | non-empty | null/absent | Continue reasoning (open thinking block). |
| 2 | non-empty | non-empty | Close reasoning, continue content. |
| 3 | null/absent | non-empty | Content-only prefill (no reasoning). |
| 4 | non-empty | `""` (empty string) | Close reasoning, generate content from scratch. |
| tool-call | (any) | (any) | Append tool tokens after reasoning/content. `content` is considered closed. |

Important rule:

- `content: ""` (empty string) closes the reasoning block and starts content
  from scratch.
- `content: null` or absent keeps the reasoning block open.

## Tool-call prefill and `__raw`

### Receiving raw tokens

A client that wants to be able to resume an assistant message should request
`return_prefill: true` (or simply save the `__raw` deltas while streaming).

Example non-streaming response snippet:

```json
{
  "message": {
    "content": "",
    "tool_calls": [{
      "id": "call_abc123",
      "type": "function",
      "function": {
        "name": "read_file",
        "arguments": "{\"path\":\"/tmp/sample.txt\"}"
      },
      "__raw": "<tool_call>\n<function=read_file>\n<parameter=path>\n/tmp/sample.txt\n</parameter>\n</function>\n</tool_call>"
    }]
  }
}
```

### Continuing with `tool_calls_raw`

To resume from a partial or complete raw call, send the saved `__raw` text in
the trailing assistant message:

```json
{
  "role": "assistant",
  "tool_calls_raw": "<tool_call>\n<function=read_file>\n<parameter=path>\n"
}
```

The model will continue the raw tokens and close the call. The response will
contain the completed structured `tool_calls` with a full `__raw` string.

### Continuing with structured `tool_calls`

For **complete** calls only, the client can send the structured array:

```json
{
  "role": "assistant",
  "tool_calls": [{
    "type": "function",
    "function": {"name": "read_file", "arguments": "{\"path\":\"/tmp/sample.txt\"}"}
  }]
}
```

The server serializes this to raw model-format tokens internally. This is not
suitable for partial calls because the client cannot represent a partial call
as structured JSON.

## Streaming behavior

With `stream: true`:

- `return_prefill: false` (default): the first deltas contain only newly
  generated tokens. The prefilled text/calls do not appear.
- `return_prefill: true`: the server emits one or more synthetic first deltas
  containing the prefilled reasoning, content, and/or complete structured tool
  calls. Subsequent deltas contain only newly generated tokens.

For tool calls, each streaming delta may contain:

- `delta.tool_calls[i].function.name` — when the function name is first known.
- `delta.tool_calls[i].function.arguments` — incremental argument fragment.
- `delta.tool_calls[i].__raw` — incremental raw token fragment.

The client should concatenate `function.arguments` per index to get the final
arguments, and concatenate `__raw` per index to get the final raw call.

## Test commands

The examples below assume a server running at `localhost:5021` with `--jinja`
and a model that supports tool calls. Adjust the model, port, and tool
definitions as needed.

Create fixtures first:

```bash
echo "hello world" > /tmp/sample.txt
echo "second file" > /tmp/sample2.txt
```

### Control: tool call without prefill

```bash
curl -s http://localhost:5021/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "system", "content": "You are a helpful assistant with access to tools."},
      {"role": "user", "content": "Read the file at /tmp/sample.txt and tell me its contents."}
    ],
    "tools": [{"type": "function", "function": {"name": "read_file", "description": "Read a file.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}}],
    "tool_choice": "auto", "max_tokens": 512, "stream": false
  }' | python3 -m json.tool
```

Expect: `finish_reason: "tool_calls"` and a populated `tool_calls[0].__raw`.

### Case 3: reasoning prefill + tool call

```bash
curl -s http://localhost:5021/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "system", "content": "You are a helpful assistant with access to tools."},
      {"role": "user", "content": "Read the file at /tmp/sample.txt and tell me its contents."},
      {"role": "assistant", "reasoning_content": "I should use the read_file tool."}
    ],
    "tools": [{"type": "function", "function": {"name": "read_file", "description": "Read a file.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}}],
    "tool_choice": "auto", "max_tokens": 512, "stream": false, "return_prefill": true
  }' | python3 -m json.tool
```

Expect: `finish_reason: "tool_calls"`, `tool_calls` populated, and
`reasoning_content` contains the prefill. With `return_prefill: false` the
prefilled reasoning is stripped.

### Case 1: reasoning-only prefill (open thinking block) + tool call

```bash
curl -s http://localhost:5021/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "system", "content": "You are a helpful assistant with access to tools."},
      {"role": "user", "content": "Read the file at /tmp/sample.txt."},
      {"role": "assistant", "reasoning_content": "I need to read the file."}
    ],
    "tools": [{"type": "function", "function": {"name": "read_file", "description": "Read a file.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}}],
    "tool_choice": "auto", "max_tokens": 512, "stream": false
  }' | python3 -m json.tool
```

Expect: `finish_reason: "tool_calls"` and `tool_calls` populated. The model
continues the open reasoning block and then emits the tool call.

### Case 2: reasoning + content prefill + tool call

```bash
curl -s http://localhost:5021/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "system", "content": "You are a helpful assistant with access to tools."},
      {"role": "user", "content": "Read the file at /tmp/sample.txt."},
      {"role": "assistant", "reasoning_content": "I need to read the file.", "content": "I will now read the file."}
    ],
    "tools": [{"type": "function", "function": {"name": "read_file", "description": "Read a file.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}}],
    "tool_choice": "auto", "max_tokens": 512, "stream": false, "return_prefill": true
  }' | python3 -m json.tool
```

Expect: `finish_reason: "tool_calls"`, `tool_calls` populated, and both
`reasoning_content` and `content` include the prefill. With `return_prefill:
false` both are stripped to the generated portion.

### Case 4: reasoning + empty content (closed reasoning) + tool call

```bash
curl -s http://localhost:5021/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "system", "content": "You are a helpful assistant with access to tools."},
      {"role": "user", "content": "Read the file at /tmp/sample.txt."},
      {"role": "assistant", "reasoning_content": "I need to read the file.", "content": ""}
    ],
    "tools": [{"type": "function", "function": {"name": "read_file", "description": "Read a file.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}}],
    "tool_choice": "auto", "max_tokens": 512, "stream": false
  }' | python3 -m json.tool
```

Expect: `finish_reason: "tool_calls"` and `tool_calls` populated. The empty
`content` closes the reasoning block; with `return_prefill: false` the
reasoning content is stripped.

### Case 3: content-only prefill

```bash
curl -s http://localhost:5021/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "system", "content": "You are a helpful assistant. Answer concisely."},
      {"role": "user", "content": "What is 2+2?"},
      {"role": "assistant", "content": "The answer is"}
    ],
    "max_tokens": 32, "stream": false, "return_prefill": true
  }' | python3 -m json.tool
```

Expect: `content: "The answer is 4."`. With `return_prefill: false` the
response is `content: " 4."`.

### Complete structured `tool_calls` prefill

```bash
curl -s http://localhost:5021/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "system", "content": "You are a helpful assistant with access to tools."},
      {"role": "user", "content": "Read the file at /tmp/sample.txt."},
      {"role": "assistant", "tool_calls": [{"type": "function", "function": {"name": "read_file", "arguments": "{\"path\":\"/tmp/sample.txt\"}"}}]}
    ],
    "tools": [{"type": "function", "function": {"name": "read_file", "description": "Read a file.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}}],
    "tool_choice": "auto", "max_tokens": 256, "stream": false, "return_prefill": true
  }' | python3 -m json.tool
```

Expect: `tool_calls` contains the prefilled call. With `return_prefill: false`
the prefilled call is stripped; if the model generated nothing new the response
has 0 tool calls and `finish_reason: "stop"`.

### Complete raw `tool_calls_raw` prefill

```bash
curl -s http://localhost:5021/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "system", "content": "You are a helpful assistant with access to tools."},
      {"role": "user", "content": "Read the file at /tmp/sample.txt."},
      {"role": "assistant", "tool_calls_raw": "<tool_call>\n<function=read_file>\n<parameter=path>\n/tmp/sample.txt\n</parameter>\n</function>\n</tool_call>"}
    ],
    "tools": [{"type": "function", "function": {"name": "read_file", "description": "Read a file.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}}],
    "tool_choice": "auto", "max_tokens": 256, "stream": false, "return_prefill": true
  }' | python3 -m json.tool
```

Expect: `tool_calls` contains the prefilled call with matching `__raw`.

### Partial raw `tool_calls_raw` prefill

Use a prefix of the raw call. The model should complete it.

```bash
curl -s http://localhost:5021/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "system", "content": "You are a helpful assistant with access to tools."},
      {"role": "user", "content": "Read the file at /tmp/sample.txt."},
      {"role": "assistant", "tool_calls_raw": "<tool_call>\n<function=read_file>\n<parameter=path>\n"}
    ],
    "tools": [{"type": "function", "function": {"name": "read_file", "description": "Read a file.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}}],
    "tool_choice": "auto", "max_tokens": 256, "stream": false, "return_prefill": false
  }' | python3 -m json.tool
```

Expect: `finish_reason: "tool_calls"`, one completed `read_file` call with
`arguments: {"path":"/tmp/sample.txt"}`. The response includes the completed
call even with `return_prefill: false`.

### Multiple complete prefilled tool calls

```bash
curl -s http://localhost:5021/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "system", "content": "You are a helpful assistant with access to tools."},
      {"role": "user", "content": "Read /tmp/sample.txt and /tmp/sample2.txt."},
      {"role": "assistant", "tool_calls_raw": "<tool_call>\n<function=read_file>\n<parameter=path>\n/tmp/sample.txt\n</parameter>\n</function>\n</tool_call><tool_call>\n<function=read_file>\n<parameter=path>\n/tmp/sample2.txt\n</parameter>\n</function>\n</tool_call>"}
    ],
    "tools": [{"type": "function", "function": {"name": "read_file", "description": "Read a file.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}}],
    "tool_choice": "auto", "max_tokens": 256, "stream": false, "return_prefill": true
  }' | python3 -m json.tool
```

Expect: `tool_calls` contains the two prefilled calls. With
`return_prefill: false` both are stripped; the response contains only calls
generated beyond them.

### Multiple prefilled calls with one partial

Use `tool_calls_raw` with two complete calls followed by an incomplete third
call. The model should complete the third call.

```bash
curl -s http://localhost:5021/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "system", "content": "You are a helpful assistant with access to tools."},
      {"role": "user", "content": "Read /tmp/sample.txt, /tmp/sample2.txt, and /tmp/sample.txt again."},
      {"role": "assistant", "tool_calls_raw": "<tool_call>\n<function=read_file>\n<parameter=path>\n/tmp/sample.txt\n</parameter>\n</function>\n</tool_call><tool_call>\n<function=read_file>\n<parameter=path>\n/tmp/sample2.txt\n</parameter>\n</function>\n</tool_call><tool_call>\n<function=read_file>\n<parameter=path>\n"}
    ],
    "tools": [{"type": "function", "function": {"name": "read_file", "description": "Read a file.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}}],
    "tool_choice": "auto", "max_tokens": 256, "stream": false, "return_prefill": true
  }' | python3 -m json.tool
```

Expect: `tool_calls` contains three calls. The third one is completed by the
model. With `return_prefill: false` the first two complete calls are stripped,
but the third (partial -> completed) call is kept.

### Streaming with `return_prefill: true`

```bash
curl -s http://localhost:5021/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "system", "content": "You are a helpful assistant with access to tools."},
      {"role": "user", "content": "Read the file at /tmp/sample.txt."},
      {"role": "assistant", "reasoning_content": "I should use the read_file tool."}
    ],
    "tools": [{"type": "function", "function": {"name": "read_file", "description": "Read a file.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}}],
    "tool_choice": "auto", "max_tokens": 512, "stream": true, "return_prefill": true
  }' > /tmp/stream-output.txt

python3 - <<'PY'
import json
for line in open('/tmp/stream-output.txt'):
    if not line.startswith('data: '):
        continue
    data = line[6:].strip()
    if data == '[DONE]':
        continue
    obj = json.loads(data)
    for c in obj.get('choices', []):
        d = c.get('delta', {})
        if 'reasoning_content' in d and d['reasoning_content']:
            print('reasoning:', repr(d['reasoning_content'][:80]))
        if 'content' in d and d['content']:
            print('content:', repr(d['content'][:80]))
        if 'tool_calls' in d:
            for tc in d['tool_calls']:
                print('tool_call delta:', json.dumps(tc)[:200])
        if c.get('finish_reason'):
            print('finish_reason:', c['finish_reason'])
PY
```

Expect: first deltas contain the prefilled reasoning, then tool-call deltas
appear with incremental `function.arguments` and per-delta `__raw`.

### Streaming tool-call continuation

1. Run a streaming request and save the cumulative `__raw` for each tool call.
2. Send that `__raw` back as `tool_calls_raw` in a new request.
3. Verify the model completes or continues from the prefilled call.

```bash
# Step 1: stream and extract __raw
curl -s http://localhost:5021/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "system", "content": "You are a helpful assistant with access to tools."},
      {"role": "user", "content": "Read the file at /tmp/sample.txt."}
    ],
    "tools": [{"type": "function", "function": {"name": "read_file", "description": "Read a file.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}}],
    "tool_choice": "auto", "max_tokens": 256, "stream": true
  }' > /tmp/stream-output.txt

python3 - <<'PY' > /tmp/partial_raw.txt
import json
raw_by_index = {}
for line in open('/tmp/stream-output.txt'):
    if not line.startswith('data: '):
        continue
    data = line[6:].strip()
    if data == '[DONE]':
        continue
    obj = json.loads(data)
    for c in obj.get('choices', []):
        for tc in c.get('delta', {}).get('tool_calls', []):
            idx = tc.get('index', 0)
            raw_by_index[idx] = raw_by_index.get(idx, '') + tc.get('__raw', '')
print(raw_by_index.get(0, ''), end='')
PY

# Step 2: use the saved raw as a partial prefill
curl -s http://localhost:5021/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "{
    \"messages\": [
      {\"role\": \"system\", \"content\": \"You are a helpful assistant with access to tools.\"},
      {\"role\": \"user\", \"content\": \"Read the file at /tmp/sample.txt.\"},
      {\"role\": \"assistant\", \"tool_calls_raw\": \"$(cat /tmp/partial_raw.txt)\"}
    ],
    \"tools\": [{\"type\": \"function\", \"function\": {\"name\": \"read_file\", \"description\": \"Read a file.\", \"parameters\": {\"type\": \"object\", \"properties\": {\"path\": {\"type\": \"string\"}}, \"required\": [\"path\"]}}}],
    \"tool_choice\": \"auto\", \"max_tokens\": 256, \"stream\": false
  }" | python3 -m json.tool
```

Expect: the response contains the completed tool call.

## Limitations and notes

- `tool_calls` (structured array) is allowed only for **complete** tool calls.
  Partial calls must use `tool_calls_raw`.
- If `tool_calls_raw` is present, `content` is considered closed.
- `__raw` in streaming deltas is a **delta**, not cumulative. The client must
  concatenate per index.
- Multi-tool scenarios with a partial last call keep the completed partial
  call when `return_prefill=false`, but strip any complete prefilled calls
  before it.
- Partial raw prefill disables the tool-call grammar. This works well with
  `tool_choice: auto`. With `tool_choice: required` the client should provide
  enough prefilled context to keep the model in tool-call mode.
