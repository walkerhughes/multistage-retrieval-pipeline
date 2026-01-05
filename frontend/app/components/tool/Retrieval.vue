<script setup lang="ts">
import type { RetrievalUIToolInvocation, ChunkResult } from '~~/shared/utils/tools/retrieval'

const props = defineProps<{
  invocation: RetrievalUIToolInvocation
}>()

const showSources = ref(false)
const expandedChunks = ref<Set<number>>(new Set())

function toggleChunk(index: number) {
  if (expandedChunks.value.has(index)) {
    expandedChunks.value.delete(index)
  } else {
    expandedChunks.value.add(index)
  }
}

function truncateText(text: string, maxLength: number = 200): string {
  if (text.length <= maxLength) return text
  return text.slice(0, maxLength) + '...'
}

function formatScore(score: number): string {
  return score.toFixed(3)
}

const color = computed(() => {
  return ({
    'output-available': 'bg-gradient-to-br from-emerald-500 via-teal-500 to-cyan-600 dark:from-emerald-600 dark:via-teal-600 dark:to-cyan-700 text-white',
    'output-error': 'bg-muted text-error'
  })[props.invocation.state as string] || 'bg-muted text-white'
})

const icon = computed(() => {
  return ({
    'input-available': 'i-lucide-search',
    'output-error': 'i-lucide-triangle-alert'
  })[props.invocation.state as string] || 'i-lucide-loader-circle'
})

const message = computed(() => {
  return ({
    'input-available': 'Searching transcripts...',
    'output-error': props.invocation.state === 'output-available' && props.invocation.output?.error
      ? props.invocation.output.error
      : 'Search failed, please try again'
  })[props.invocation.state as string] || 'Searching transcripts...'
})

const hasAnswer = computed(() => {
  return props.invocation.state === 'output-available' && props.invocation.output?.answer
})

const hasResults = computed(() => {
  return props.invocation.state === 'output-available' &&
    props.invocation.output?.chunks &&
    props.invocation.output.chunks.length > 0
})

const hasError = computed(() => {
  return props.invocation.state === 'output-available' && props.invocation.output?.error
})
</script>

<template>
  <div class="rounded-xl my-5 overflow-hidden" :class="hasError ? 'bg-muted' : ''">
    <!-- Header -->
    <div class="px-5 py-4" :class="color">
      <div class="flex items-center justify-between">
        <div class="flex items-center gap-2">
          <UIcon name="i-lucide-brain" class="size-5" />
          <span class="font-medium">Multi-Query Search</span>
        </div>
        <template v-if="invocation.state === 'output-available' && !hasError">
          <div class="text-sm text-white/80">
            {{ invocation.output.chunks.length }} source{{ invocation.output.chunks.length !== 1 ? 's' : '' }}
            <span v-if="invocation.output.latency_ms" class="ml-2">
              ({{ Math.round(invocation.output.latency_ms) }}ms)
            </span>
          </div>
        </template>
      </div>
      <div v-if="invocation.input?.question" class="mt-2 text-sm text-white/70">
        "{{ invocation.input.question }}"
      </div>
    </div>

    <!-- Answer and Sub-queries -->
    <template v-if="hasAnswer">
      <div class="bg-default px-5 py-4">
        <!-- Sub-queries -->
        <div v-if="invocation.output.sub_queries?.length > 0" class="mb-4">
          <div class="text-xs font-medium text-muted mb-2">Sub-queries generated:</div>
          <div class="flex flex-wrap gap-2">
            <span
              v-for="(subQuery, index) in invocation.output.sub_queries"
              :key="index"
              class="text-xs bg-muted px-2 py-1 rounded"
            >
              {{ subQuery }}
            </span>
          </div>
        </div>

        <!-- Answer -->
        <div class="prose prose-sm dark:prose-invert max-w-none">
          <p class="whitespace-pre-wrap">{{ invocation.output.answer }}</p>
        </div>

        <!-- Toggle sources -->
        <button
          v-if="hasResults"
          class="mt-4 text-sm text-primary flex items-center gap-1 hover:underline"
          @click="showSources = !showSources"
        >
          <UIcon :name="showSources ? 'i-lucide-chevron-up' : 'i-lucide-chevron-down'" class="size-4" />
          {{ showSources ? 'Hide' : 'Show' }} {{ invocation.output.chunks.length }} source{{ invocation.output.chunks.length !== 1 ? 's' : '' }}
        </button>
      </div>

      <!-- Source chunks (collapsible) -->
      <div v-if="showSources && hasResults" class="bg-muted/30 divide-y divide-default max-h-64 overflow-y-auto">
        <div
          v-for="(chunk, index) in (invocation.output.chunks as ChunkResult[])"
          :key="chunk.chunk_id"
          class="px-5 py-3 hover:bg-muted/50 cursor-pointer transition-colors"
          @click="toggleChunk(index)"
        >
          <div class="flex items-start justify-between gap-3">
            <div class="flex-1 min-w-0">
              <div class="flex items-center gap-2 mb-1">
                <span class="text-xs font-medium text-muted bg-muted px-2 py-0.5 rounded">
                  #{{ index + 1 }}
                </span>
                <span v-if="chunk.metadata?.title" class="text-sm font-medium truncate">
                  {{ chunk.metadata.title }}
                </span>
                <span class="text-xs text-muted">
                  Score: {{ formatScore(chunk.score) }}
                </span>
              </div>
              <p class="text-sm text-muted leading-relaxed">
                {{ expandedChunks.has(index) ? chunk.text : truncateText(chunk.text) }}
              </p>
            </div>
            <UIcon
              :name="expandedChunks.has(index) ? 'i-lucide-chevron-up' : 'i-lucide-chevron-down'"
              class="size-4 text-muted flex-shrink-0 mt-1"
            />
          </div>
        </div>
      </div>
    </template>

    <!-- Loading state -->
    <div v-else-if="invocation.state !== 'output-available'" class="flex items-center justify-center h-32 bg-muted">
      <div class="text-center">
        <UIcon
          :name="icon"
          class="size-8 mx-auto mb-2"
          :class="[invocation.state === 'input-streaming' && 'animate-spin']"
        />
        <div class="text-sm">
          {{ message }}
        </div>
      </div>
    </div>

    <!-- Error or no results -->
    <div v-else-if="hasError || !hasAnswer" class="flex items-center justify-center h-32 bg-muted">
      <div class="text-center">
        <UIcon
          :name="hasError ? 'i-lucide-triangle-alert' : 'i-lucide-search-x'"
          class="size-8 mx-auto mb-2 text-muted"
        />
        <div class="text-sm text-muted">
          {{ hasError ? invocation.output.error : 'No matching transcripts found' }}
        </div>
      </div>
    </div>
  </div>
</template>
