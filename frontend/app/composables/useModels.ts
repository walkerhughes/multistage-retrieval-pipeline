export function formatModelName(modelId: string): string {
  const acronyms = ['gpt'] // words that should be uppercase
  const modelName = modelId.split('/')[1] || modelId

  return modelName
    .split('-')
    .map((word) => {
      const lowerWord = word.toLowerCase()
      return acronyms.includes(lowerWord)
        ? word.toUpperCase()
        : word.charAt(0).toUpperCase() + word.slice(1)
    })
    .join(' ')
}

export function useModels() {
  const models = [
    'openai/gpt-4o-mini',
    'openai/gpt-5-nano'
  ]

  const model = useCookie<string>('model', { default: () => 'openai/gpt-5-nano' })

  return {
    models,
    model,
    formatModelName
  }
}
