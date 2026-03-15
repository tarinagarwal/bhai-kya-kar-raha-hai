/**
 * Gemini Client — Vertex AI / Gemini API wrapper for ClawdCursor.
 *
 * Provides:
 *  - Text-only calls (task decomposition, Layer 2 planning)
 *  - Vision calls (screenshot → action JSON, Layer 3)
 *  - Conversation-based vision loop for computer use
 *
 * Auth: Vertex AI uses Application Default Credentials (gcloud CLI).
 *       Gemini API uses an API key.
 */

import { GoogleGenAI, type Content, type Part } from "@google/genai";

export interface GeminiConfig {
  project: string;
  location: string;
  model: string;
  useVertexAI: boolean;
  apiKey?: string;
}

let _client: GoogleGenAI | null = null;
let _lastConfig: string = "";

/**
 * Get or create a singleton GoogleGenAI client.
 * Reuses the client if config hasn't changed.
 */
export function getGeminiClient(config: GeminiConfig): GoogleGenAI {
  const configKey = JSON.stringify(config);
  if (_client && _lastConfig === configKey) return _client;

  if (config.useVertexAI) {
    _client = new GoogleGenAI({
      vertexai: true,
      project: config.project,
      location: config.location,
    });
  } else if (config.apiKey) {
    _client = new GoogleGenAI({ apiKey: config.apiKey });
  } else {
    throw new Error(
      "Gemini client requires either Vertex AI config or an API key",
    );
  }

  _lastConfig = configKey;
  return _client;
}

/**
 * Build Gemini config from environment variables.
 */
export function buildGeminiConfig(
  overrides?: Partial<GeminiConfig>,
): GeminiConfig {
  return {
    useVertexAI: overrides?.useVertexAI ?? process.env.USE_VERTEXAI === "true",
    project: overrides?.project ?? process.env.VERTEXAI_PROJECT ?? "",
    location: overrides?.location ?? process.env.VERTEXAI_LOCATION ?? "global",
    model:
      overrides?.model ?? process.env.GEMINI_MODEL ?? "gemini-3-flash-preview",
    apiKey: overrides?.apiKey ?? process.env.GEMINI_API_KEY,
  };
}

/**
 * Text-only generation — used for task decomposition and planning.
 */
export async function geminiTextCall(
  config: GeminiConfig,
  systemPrompt: string,
  userMessage: string,
  maxTokens: number = 1024,
): Promise<string> {
  const client = getGeminiClient(config);

  for (let attempt = 0; attempt < 3; attempt++) {
    try {
      const response = await client.models.generateContent({
        model: config.model,
        config: {
          maxOutputTokens: maxTokens,
          systemInstruction: systemPrompt,
        },
        contents: [{ role: "user", parts: [{ text: userMessage }] }],
      });

      return response.text ?? "";
    } catch (err: any) {
      const errStr = String(err);
      if (
        (errStr.includes("429") || errStr.includes("RESOURCE_EXHAUSTED")) &&
        attempt < 2
      ) {
        const backoff = 3000 * Math.pow(2, attempt) + Math.random() * 1000;
        console.log(
          `   ⏳ Rate limited — retrying in ${Math.round(backoff / 1000)}s...`,
        );
        await new Promise((r) => setTimeout(r, backoff));
        continue;
      }
      throw err;
    }
  }
  throw new Error("geminiTextCall: max retries exceeded");
}

/**
 * Vision generation — send screenshot + text prompt, get text response.
 * Supports multi-turn conversation via the messages array.
 */
export async function geminiVisionCall(
  config: GeminiConfig,
  systemPrompt: string,
  messages: GeminiMessage[],
  maxTokens: number = 2048,
): Promise<string> {
  const client = getGeminiClient(config);

  // Convert our message format to Gemini Content format
  const contents: Content[] = messages.map((msg) => ({
    role: msg.role === "assistant" ? "model" : "user",
    parts: msg.parts,
  }));

  for (let attempt = 0; attempt < 3; attempt++) {
    try {
      const response = await client.models.generateContent({
        model: config.model,
        config: {
          maxOutputTokens: maxTokens,
          systemInstruction: systemPrompt,
        },
        contents,
      });

      return response.text ?? "";
    } catch (err: any) {
      const errStr = String(err);
      if (
        (errStr.includes("429") || errStr.includes("RESOURCE_EXHAUSTED")) &&
        attempt < 2
      ) {
        const backoff = 5000 * Math.pow(2, attempt) + Math.random() * 2000;
        console.log(
          `   ⏳ Rate limited — retrying in ${Math.round(backoff / 1000)}s...`,
        );
        await new Promise((r) => setTimeout(r, backoff));
        continue;
      }
      throw err;
    }
  }
  throw new Error("geminiVisionCall: max retries exceeded");
}

export interface GeminiMessage {
  role: "user" | "assistant";
  parts: Part[];
}

/**
 * Create an inline image part from a Buffer.
 */
export function createImagePart(
  buffer: Buffer,
  format: "png" | "jpeg" | "raw",
): Part {
  return {
    inlineData: {
      mimeType: format === "jpeg" ? "image/jpeg" : "image/png",
      data: buffer.toString("base64"),
    },
  };
}

/**
 * Create a text part.
 */
export function createTextPart(text: string): Part {
  return { text };
}

/**
 * Test if Gemini is reachable with the given config.
 */
export async function testGeminiConnection(
  config: GeminiConfig,
): Promise<{ ok: boolean; latencyMs?: number; error?: string }> {
  const start = performance.now();
  try {
    const result = await geminiTextCall(config, "", "Reply OK", 5);
    if (!result) return { ok: false, error: "Empty response" };
    return { ok: true, latencyMs: Math.round(performance.now() - start) };
  } catch (err: any) {
    return { ok: false, error: err.message || String(err) };
  }
}
