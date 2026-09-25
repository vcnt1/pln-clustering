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
  if (score === null) return 'Sem dados de humor ainda'
  const normalizedScore = normalizeScore(score, scale)
  if (normalizedScore <= -0.6) return 'Muito negativo'
  if (normalizedScore <= -0.2) return 'Negativo'
  if (normalizedScore < 0.2) return 'Neutro'
  if (normalizedScore < 0.6) return 'Positivo'
  return 'Muito positivo'
}