import { useState } from 'react'
import type { FormEvent, KeyboardEvent } from 'react'
import './App.css'
import { getCustomerMood, sendMessage } from './services/api'
import { getMoodEmoji, getMoodLabel } from './utils/mood'

type Role = 'agent' | 'customer'
type Message = { id: string; role: Role; text: string; time: string }
type Conversation = {
  id: string; customerId: string; customerName: string; initials: string; room: string
  score: number | null; scale: string; lastSeen: string; messages: Message[]
}

// A new persona only carries a generated id; nothing here is invented data.
function createConversation(): Conversation {
  const id = crypto.randomUUID()
  const shortId = id.slice(0, 4).toUpperCase()
  return {
    id: `conv-${id}`,
    customerId: `customer-${id}`,
    customerName: `Customer ${shortId}`,
    initials: shortId.slice(0, 2),
    room: '',
    score: null,
    scale: '-1 to 1',
    lastSeen: '',
    messages: [],
  }
}

function App() {
  const [conversations, setConversations] = useState<Conversation[]>([])
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [draft, setDraft] = useState('')
  const [sendRole, setSendRole] = useState<Role>('agent')
  const selectedConversation = conversations.find(({ id }) => id === selectedId) ?? null

  function handleAddConversation() {
    const conversation = createConversation()
    setConversations((current) => [conversation, ...current])
    setSelectedId(conversation.id)
  }

  async function handleSend(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    if (!selectedConversation) return
    const text = draft.trim()
    if (!text) return
    const message = await sendMessage({ conversationId: selectedConversation.id, customerId: selectedConversation.customerId, role: sendRole, message: text })
    setConversations((current) => current.map((conversation) => conversation.id === selectedConversation.id
      ? { ...conversation, messages: [...conversation.messages, message], lastSeen: message.time } : conversation))
    setDraft('')
    // Only a customer-role message can move the score (ADR-0007); refresh right
    // after it since mood-api computes inference synchronously during ingest.
    if (sendRole === 'customer') {
      await refreshMood()
    }
  }

  function handleComposerKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault()
      event.currentTarget.form?.requestSubmit()
    }
  }

  async function refreshMood() {
    if (!selectedConversation) return
    const conversationId = selectedConversation.id
    const mood = await getCustomerMood(selectedConversation.customerId)
    setConversations((current) => current.map((conversation) => {
      if (conversation.id !== conversationId) return conversation
      // data-model.md §7: only apply the score to the conversation it belongs to.
      if (mood === null || mood.conversation_id !== conversationId) return { ...conversation, score: null }
      return { ...conversation, score: mood.score, scale: mood.scale }
    }))
  }

  return (
    <main className="app-shell">
      <aside className="sidebar">
        <div className="brand-row"><div className="brand-mark">H</div><div><p className="eyebrow">Front desk</p><h1>Haven desk</h1></div><button className="icon-button" type="button" aria-label="Open workspace menu">•••</button></div>
        <div className="inbox-heading"><div><p className="eyebrow">Your inbox</p><h2>Conversations <span>{conversations.length}</span></h2></div><button className="compose-button" type="button" aria-label="Start a new conversation" onClick={handleAddConversation}>+</button></div>
        <label className="search-box"><span aria-hidden="true">⌕</span><input type="search" placeholder="Search guests" aria-label="Search guests" /><kbd>/</kbd></label>
        <nav className="conversation-list" aria-label="Conversations">
          {conversations.length === 0 && <p className="empty-hint">No conversations yet. Click + to add one.</p>}
          {conversations.map((conversation) => (
            <button className={`conversation-item ${conversation.id === selectedId ? 'is-selected' : ''}`} key={conversation.id} type="button" onClick={() => setSelectedId(conversation.id)}>
              <span className="avatar">{conversation.initials}</span>
              <span className="conversation-copy">
                <span className="conversation-name">{getMoodEmoji(conversation.score, conversation.scale)} {conversation.customerName}</span>
                <span className="conversation-preview">{conversation.messages.at(-1)?.text ?? 'No messages yet'}</span>
              </span>
              <span className="conversation-time">{conversation.lastSeen}</span>
            </button>
          ))}
        </nav>
        <div className="sidebar-footer"><div className="profile-avatar">AM</div><div><strong>Alex Morgan</strong><span>On duty · Concierge</span></div><button className="icon-button" type="button" aria-label="Open profile settings">⚙</button></div>
      </aside>
      <section className="chat-panel" aria-label="Selected conversation">
        {selectedConversation ? (
          <>
            <header className="chat-header">
              <div className="guest-heading">
                <div className="avatar avatar-large">{selectedConversation.initials}</div>
                <div>
                  <div className="name-line"><h2>{selectedConversation.customerName}</h2><span className="online-dot" /><span className="online-label">Online</span></div>
                  <p>{selectedConversation.room || 'No room assigned yet'}</p>
                </div>
              </div>
              <div className="header-actions"><button className="secondary-button" type="button" onClick={refreshMood}><span>↻</span> Refresh mood</button><button className="icon-button outlined" type="button" aria-label="More conversation actions">•••</button></div>
            </header>
            <div className="mood-strip">
              <div className="mood-icon">{getMoodEmoji(selectedConversation.score, selectedConversation.scale)}</div>
              <div><span className="eyebrow">Current guest mood</span><strong>{getMoodLabel(selectedConversation.score, selectedConversation.scale)}</strong></div>
              <div className="mood-meter" aria-label={`Mood score ${selectedConversation.score ?? 'unavailable'}`}><span style={{ width: selectedConversation.score === null ? '0%' : `${((selectedConversation.score + 1) / 2) * 100}%` }} /></div>
              <span className="mood-score">{selectedConversation.score === null ? '—' : `${selectedConversation.score > 0 ? '+' : ''}${selectedConversation.score.toFixed(2)}`}</span>
              <span className="model-tag">ML v0.1</span>
            </div>
            <div className="message-area">
              {selectedConversation.messages.length === 0 ? (
                <p className="empty-hint">No messages yet. Say hello to get the conversation started.</p>
              ) : (
                <>
                  <div className="date-divider"><span>Today</span></div>
                  <div className="messages">
                    {selectedConversation.messages.map((message) => (
                      <div className={`message-row ${message.role}`} key={message.id}>
                        {message.role === 'customer' && <span className="avatar avatar-small">{selectedConversation.initials}</span>}
                        <div className="message-content">
                          <div className="message-bubble">{message.text}</div>
                          <span className="message-time">{message.time}{message.role === 'agent' && '  ·  Read'}</span>
                        </div>
                      </div>
                    ))}
                  </div>
                </>
              )}
            </div>
            <div className="composer-role-toggle" role="radiogroup" aria-label="Send message as">
              <button type="button" className={sendRole === 'agent' ? 'is-active' : ''} onClick={() => setSendRole('agent')} aria-pressed={sendRole === 'agent'}>Agent reply</button>
              <button type="button" className={sendRole === 'customer' ? 'is-active' : ''} onClick={() => setSendRole('customer')} aria-pressed={sendRole === 'customer'}>Simulate customer (test)</button>
            </div>
            <form className="composer" onSubmit={handleSend}>
              <textarea value={draft} onChange={(event) => setDraft(event.target.value)} onKeyDown={handleComposerKeyDown} placeholder="Write a reply..." aria-label="Message" rows={1} />
              <div className="composer-actions"><span className="composer-hint">Enter to send · Shift+Enter for newline</span><button className="send-button" type="submit" disabled={!draft.trim()} aria-label="Send message">↑</button></div>
            </form>
          </>
        ) : (
          <div className="empty-panel"><p>Select a conversation or click + to add a new customer.</p></div>
        )}
      </section>
    </main>
  )
}

export default App

