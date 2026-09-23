export type MoodResponse = {
  customer_id: string
  conversation_id: string
  score: number
  scale: string
  mood_label: string | null
  model_version: string
  computed_at: string
}
type SendMessageInput = { conversationId: string; customerId: string; role: 'agent' | 'customer'; message: string }
type SentMessage = { id: string; role: SendMessageInput['role']; text: string; time: string }

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000'

export async function sendMessage(input: SendMessageInput): Promise<SentMessage> {
  const timestamp = new Date().toISOString()
  const response = await fetch(`${API_BASE_URL}/v1alpha1/ingest`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      conversation_id: input.conversationId,
      customer_id: input.customerId,
      role: input.role,
      message: input.message,
      timestamp,
    }),
  })
  if (!response.ok) {
    const body = await response.text()
    throw new Error(`ingest failed (${response.status}): ${body}`)
  }
  const data = await response.json() as { message_id: string }
  return {
    id: data.message_id,
    role: input.role,
    text: input.message,
    time: new Date(timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
  }
}

export async function getCustomerMood(customerId: string): Promise<MoodResponse | null> {
  const response = await fetch(`${API_BASE_URL}/v1alpha1/customer/${encodeURIComponent(customerId)}/mood`)
  if (response.status === 404) {
    return null
  }
  if (!response.ok) {
    const body = await response.text()
    throw new Error(`mood fetch failed (${response.status}): ${body}`)
  }
  return response.json() as Promise<MoodResponse>
}