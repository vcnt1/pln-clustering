function normalizeScore(score: number, scale: string): number {
  return scale === '0 to 1' ? (score * 2) - 1 : score
}

export function getMoodEmoji(score: number, scale: string): string {
  const normalizedScore = normalizeScore(score, scale)
  if (normalizedScore <= -0.6) return '😠'
  if (normalizedScore < -0.1) return '🙁'
  if (normalizedScore <= 0.35) return '😐'
  if (normalizedScore <= 0.7) return '🙂'
  return '😄'
}

export function getMoodLabel(score: number, scale: string): string {
  const normalizedScore = normalizeScore(score, scale)
  if (normalizedScore <= -0.6) return 'Very negative'
  if (normalizedScore < -0.1) return 'Negative'
  if (normalizedScore <= 0.35) return 'Neutral'
  if (normalizedScore <= 0.7) return 'Positive'
  return 'Very positive'
}