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
export type ConversationMessageDto = { message_id: string; role: 'agent' | 'customer'; text: string; sent_at: string }
export type ConversationSummaryDto = {
  conversation_id: string
  customer_id: string
  last_message_at: string
  score: number | null
  scale: string | null
  messages: ConversationMessageDto[]
}

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000'

export function formatTime(iso: string): string {
  return new Date(iso).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
}

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
    time: formatTime(timestamp),
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

export async function getConversations(): Promise<ConversationSummaryDto[]> {
  const response = await fetch(`${API_BASE_URL}/v1alpha1/conversations`)
  if (!response.ok) {
    const body = await response.text()
    throw new Error(`conversations fetch failed (${response.status}): ${body}`)
  }
  return response.json() as Promise<ConversationSummaryDto[]>
}

export async function deleteConversation(conversationId: string): Promise<void> {
  const response = await fetch(`${API_BASE_URL}/v1alpha1/conversations/${encodeURIComponent(conversationId)}`, {
    method: 'DELETE',
  })
  if (!response.ok && response.status !== 404) {
    const body = await response.text()
    throw new Error(`delete conversation failed (${response.status}): ${body}`)
  }
}