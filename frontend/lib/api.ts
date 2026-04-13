/**
 * API client for the DeepReads recommendation backend.
 *
 * Wraps all FastAPI endpoints with typed fetch helpers.
 *
 * AI Attribution: Generated with assistance from Claude (Anthropic).
 */

import type {
  CompareRequest,
  CompareResponse,
  HealthResponse,
  Persona,
  PopularItem,
  RecommendRequest,
  RecommendResponse,
} from "./types";

const BASE_URL =
  process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8001";

async function request<T>(
  path: string,
  options?: RequestInit,
): Promise<T> {
  const url = `${BASE_URL}${path}`;
  const res = await fetch(url, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...options?.headers,
    },
  });
  if (!res.ok) {
    const text = await res.text().catch(() => "Unknown error");
    throw new Error(`API ${res.status}: ${text}`);
  }
  return res.json() as Promise<T>;
}

export const api = {
  /** Health check. */
  health(): Promise<HealthResponse> {
    return request<HealthResponse>("/health");
  },

  /** Get recommendations from a single model. */
  recommend(body: RecommendRequest): Promise<RecommendResponse> {
    return request<RecommendResponse>("/recommend", {
      method: "POST",
      body: JSON.stringify(body),
    });
  },

  /** Compare all three models side-by-side. */
  compare(body: CompareRequest): Promise<CompareResponse> {
    return request<CompareResponse>("/compare", {
      method: "POST",
      body: JSON.stringify(body),
    });
  },

  /** Fetch popular items for the item picker. */
  popularItems(n = 50): Promise<PopularItem[]> {
    return request<PopularItem[]>(`/items/popular?n=${n}`);
  },

  /** Fetch demo personas. */
  personas(): Promise<Persona[]> {
    return request<Persona[]>("/personas");
  },
};
