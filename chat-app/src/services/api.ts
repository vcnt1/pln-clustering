export type MoodResponse = { customer_id: string; score: number; scale: string; model_version: string; computed_at: string }
type SendMessageInput = { conversationId: string; customerId: string; role: 'agent' | 'customer'; message: string }
type SentMessage = { id: string; role: SendMessageInput['role']; text: string; time: string }

export async function sendMessage(input: SendMessageInput): Promise<SentMessage> {
  void input.conversationId
  void input.customerId
  return { id: `local-${Date.now()}`, role: input.role, text: input.message, time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) }
}

export async function getCustomerMood(customerId: string): Promise<MoodResponse> {
  return { customer_id: customerId, score: 0.78, scale: '-1 to 1', model_version: 'mock-v0.1', computed_at: new Date().toISOString() }
}