# ============================================================
# DeepReads — Next.js frontend for HuggingFace Spaces (Docker)
# Port: 7860 (HF Spaces requirement)
# ============================================================

FROM node:20-alpine

WORKDIR /app

# Install dependencies
COPY package.json package-lock.json* ./
RUN npm ci

# Copy source
COPY . .

# Build — NEXT_PUBLIC_API_URL must be set as a Space variable
# (it's baked into the JS bundle at build time)
ARG NEXT_PUBLIC_API_URL=https://yashb18-deepreads-api.hf.space
ENV NEXT_PUBLIC_API_URL=$NEXT_PUBLIC_API_URL

RUN npm run build

# HF Spaces requires port 7860
ENV PORT=7860
EXPOSE 7860

CMD ["npm", "start"]
