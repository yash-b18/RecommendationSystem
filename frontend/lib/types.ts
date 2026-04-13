/**
 * Shared TypeScript types for the DeepReads frontend.
 *
 * Mirrors the Pydantic schemas defined in src/api/schemas.py.
 *
 * AI Attribution: Generated with assistance from Claude (Anthropic).
 */

export interface RecommendedItem {
  item_idx: number;
  title: string;
  category: string;
  brand: string;
  price: number | null;
  avg_rating: number | null;
  num_ratings: number | null;
  score: number;
  explanation: string;
}

export interface RecommendRequest {
  user_idx: number | null;
  liked_items: number[];
  model: "naive" | "classical" | "deep";
  top_k: number;
}

export interface RecommendResponse {
  model_used: string;
  user_idx: number | null;
  recommendations: RecommendedItem[];
}

export interface CompareRequest {
  user_idx: number | null;
  liked_items: number[];
  top_k: number;
}

export interface CompareResponse {
  user_idx: number | null;
  naive: RecommendedItem[];
  classical: RecommendedItem[];
  deep: RecommendedItem[];
}

export interface PopularItem {
  item_idx: number;
  title: string;
  category: string;
  brand: string;
  price: number | null;
  avg_rating: number | null;
  num_ratings: number | null;
}

export interface Persona {
  persona_id: number;
  name: string;
  description: string;
  liked_item_idxs: number[];
  user_idx: number | null;
}

export interface HealthResponse {
  status: string;
  models_loaded: Record<string, boolean>;
}
