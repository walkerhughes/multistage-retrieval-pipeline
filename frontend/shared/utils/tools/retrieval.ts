import type { UIToolInvocation } from 'ai'
import { tool } from 'ai'
import { z } from 'zod'

export type RetrievalUIToolInvocation = UIToolInvocation<typeof retrievalTool>

export interface ChunkResult {
  chunk_id: number
  doc_id: number
  score: number
  text: string
  metadata: {
    title?: string
    source?: string
    published_at?: string
    [key: string]: unknown
  }
  ord: number
}

export interface RetrievalOutput {
  answer: string
  chunks: ChunkResult[]
  sub_queries: string[]
  latency_ms: number
  query: string
  error?: string
}

export const retrievalTool = tool({
  description: 'Search podcast transcripts and answer questions using multi-query retrieval. Use this tool when the user asks questions about podcast content, wants to find specific topics discussed, or needs information from video transcripts. The tool decomposes complex questions into sub-queries for better results.',
  inputSchema: z.object({
    question: z.string().describe('The question to answer using podcast transcripts'),
    limit: z.number().optional().default(10).describe('Maximum number of chunks to retrieve (1-50)')
  }),
  execute: async ({ question, limit = 10 }): Promise<RetrievalOutput> => {
    try {
      const response = await fetch('http://localhost:8000/api/chat/completion', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          question,
          agent: 'multi-query',
          mode: 'hybrid',
          operator: 'or',
          fts_candidates: 100,
          max_returned: Math.min(Math.max(limit, 1), 50)
        })
      })

      if (!response.ok) {
        const errorText = await response.text()
        return {
          answer: '',
          chunks: [],
          sub_queries: [],
          latency_ms: 0,
          query: question,
          error: `Retrieval failed: ${response.status} ${errorText}`
        }
      }

      const data = await response.json()
      return {
        answer: data.answer || '',
        chunks: data.retrieved_chunks || [],
        sub_queries: data.sub_queries || [],
        latency_ms: data.latency_ms || 0,
        query: question
      }
    } catch (error) {
      return {
        answer: '',
        chunks: [],
        sub_queries: [],
        latency_ms: 0,
        query: question,
        error: `Failed to connect to retrieval service: ${error instanceof Error ? error.message : 'Unknown error'}`
      }
    }
  }
})
