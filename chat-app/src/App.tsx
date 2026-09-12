import { useState } from 'react'
import type { FormEvent } from 'react'
import './App.css'
import { getCustomerMood, sendMessage } from './services/api'
import { getMoodEmoji, getMoodLabel } from './utils/mood'

type Role = 'agent' | 'customer'
type Message = { id: string; role: Role; text: string; time: string }
type Conversation = {
  id: string; customerId: string; customerName: string; initials: string; room: string
  score: number; scale: string; lastSeen: string; messages: Message[]
}

const initialConversations: Conversation[] = [
  { id: 'conv-1024', customerId: 'customer-1024', customerName: 'Camila Rocha', initials: 'CR', room: 'Suite 304 · Check-in today', score: 0.78, scale: '0 to 1', lastSeen: '09:42', messages: [
    { id: 'm-1', role: 'customer', text: 'Hi! Is it possible to check in a little earlier today?', time: '09:36' },
    { id: 'm-2', role: 'agent', text: 'Hello Camila! I am checking that for you now. Your room is almost ready.', time: '09:38' },
    { id: 'm-3', role: 'customer', text: 'That would be wonderful, thank you for the quick help.', time: '09:42' },
  ] },
  { id: 'conv-1025', customerId: 'customer-1025', customerName: 'Lucas Mendes', initials: 'LM', room: 'Room 817 · Staying until Friday', score: -0.42, scale: '-1 to 1', lastSeen: '08:58', messages: [
    { id: 'm-4', role: 'customer', text: 'The air conditioning is making a strange noise again.', time: '08:51' },
    { id: 'm-5', role: 'agent', text: 'I am sorry about that, Lucas. I have alerted our maintenance team.', time: '08:58' },
  ] },
  { id: 'conv-1026', customerId: 'customer-1026', customerName: 'Sofia Almeida', initials: 'SA', room: 'Room 212 · Check-out tomorrow', score: 0.08, scale: '-1 to 1', lastSeen: 'Yesterday', messages: [
    { id: 'm-6', role: 'customer', text: 'Could you send me the restaurant menu for dinner?', time: 'Yesterday' },
    { id: 'm-7', role: 'agent', text: 'Of course! I have attached the menu here. Our kitchen is open until 22:30.', time: 'Yesterday' },
  ] },
]

function App() {
  const [conversations, setConversations] = useState(initialConversations)
  const [selectedId, setSelectedId] = useState(initialConversations[0].id)
  const [draft, setDraft] = useState('')
  const selectedConversation = conversations.find(({ id }) => id === selectedId) ?? conversations[0]

  async function handleSend(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    const text = draft.trim()
    if (!text) return
    const message = await sendMessage({ conversationId: selectedConversation.id, customerId: selectedConversation.customerId, role: 'agent', message: text })
    setConversations((current) => current.map((conversation) => conversation.id === selectedConversation.id
      ? { ...conversation, messages: [...conversation.messages, message], lastSeen: message.time } : conversation))
    setDraft('')
  }

  async function refreshMood() {
    const mood = await getCustomerMood(selectedConversation.customerId)
    setConversations((current) => current.map((conversation) => conversation.customerId === mood.customer_id
      ? { ...conversation, score: mood.score, scale: mood.scale } : conversation))
  }

  return (
    <main className="app-shell">
      <aside className="sidebar">
        <div className="brand-row"><div className="brand-mark">H</div><div><p className="eyebrow">Front desk</p><h1>Haven desk</h1></div><button className="icon-button" type="button" aria-label="Open workspace menu">•••</button></div>
        <div className="inbox-heading"><div><p className="eyebrow">Your inbox</p><h2>Conversations <span>{conversations.length}</span></h2></div><button className="compose-button" type="button" aria-label="Start a new conversation">+</button></div>
        <label className="search-box"><span aria-hidden="true">⌕</span><input type="search" placeholder="Search guests" aria-label="Search guests" /><kbd>/</kbd></label>
        <nav className="conversation-list" aria-label="Conversations">{conversations.map((conversation) => <button className={`conversation-item ${conversation.id === selectedId ? 'is-selected' : ''}`} key={conversation.id} type="button" onClick={() => setSelectedId(conversation.id)}><span className="avatar">{conversation.initials}</span><span className="conversation-copy"><span className="conversation-name">{getMoodEmoji(conversation.score, conversation.scale)} {conversation.customerName}</span><span className="conversation-preview">{conversation.messages.at(-1)?.text}</span></span><span className="conversation-time">{conversation.lastSeen}</span></button>)}</nav>
        <div className="sidebar-footer"><div className="profile-avatar">AM</div><div><strong>Alex Morgan</strong><span>On duty · Concierge</span></div><button className="icon-button" type="button" aria-label="Open profile settings">⚙</button></div>
      </aside>
      <section className="chat-panel" aria-label="Selected conversation">
        <header className="chat-header"><div className="guest-heading"><div className="avatar avatar-large">{selectedConversation.initials}</div><div><div className="name-line"><h2>{selectedConversation.customerName}</h2><span className="online-dot" /><span className="online-label">Online</span></div><p>{selectedConversation.room}</p></div></div><div className="header-actions"><button className="secondary-button" type="button" onClick={refreshMood}><span>↻</span> Refresh mood</button><button className="icon-button outlined" type="button" aria-label="More conversation actions">•••</button></div></header>
        <div className="mood-strip"><div className="mood-icon">{getMoodEmoji(selectedConversation.score, selectedConversation.scale)}</div><div><span className="eyebrow">Current guest mood</span><strong>{getMoodLabel(selectedConversation.score, selectedConversation.scale)}</strong></div><div className="mood-meter" aria-label={`Mood score ${selectedConversation.score}`}><span style={{ width: `${((selectedConversation.score + 1) / 2) * 100}%` }} /></div><span className="mood-score">{selectedConversation.score > 0 ? '+' : ''}{selectedConversation.score.toFixed(2)}</span><span className="model-tag">ML v0.1</span></div>
        <div className="message-area"><div className="date-divider"><span>Today</span></div><div className="messages">{selectedConversation.messages.map((message) => <div className={`message-row ${message.role}`} key={message.id}>{message.role === 'customer' && <span className="avatar avatar-small">{selectedConversation.initials}</span>}<div className="message-content"><div className="message-bubble">{message.text}</div><span className="message-time">{message.time}{message.role === 'agent' && '  ·  Read'}</span></div></div>)}</div></div>
        <form className="composer" onSubmit={handleSend}><textarea value={draft} onChange={(event) => setDraft(event.target.value)} placeholder="Write a reply..." aria-label="Message" rows={1} /><div className="composer-actions"><span className="composer-hint">⌘ Enter to send</span><button className="send-button" type="submit" disabled={!draft.trim()} aria-label="Send message">↑</button></div></form>
      </section>
    </main>
  )
}

export default App
