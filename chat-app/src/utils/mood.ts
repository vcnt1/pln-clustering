function normalizeScore(score: number, scale: string): number {
  return scale === '0 to 1' ? (score * 2) - 1 : score
}

export function getMoodEmoji(score: number | null, scale: string): string {
  if (score === null) return '❔'
  const normalizedScore = normalizeScore(score, scale)
  if (normalizedScore <= -0.6) return '😠'
  if (normalizedScore <= -0.2) return '🙁'
  if (normalizedScore < 0.2) return '😐'
  if (normalizedScore < 0.6) return '🙂'
  return '😄'
}

export function getMoodLabel(score: number | null, scale: string): string {
  if (score === null) return 'No mood data yet'
  const normalizedScore = normalizeScore(score, scale)
  if (normalizedScore <= -0.6) return 'Very negative'
  if (normalizedScore <= -0.2) return 'Negative'
  if (normalizedScore < 0.2) return 'Neutral'
  if (normalizedScore < 0.6) return 'Positive'
  return 'Very positive'
}