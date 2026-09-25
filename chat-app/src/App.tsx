import { useEffect, useRef, useState } from 'react'
import type { FormEvent, KeyboardEvent } from 'react'
import './App.css'
import type { ConversationSummaryDto } from './services/api'
import { deleteConversation, formatTime, getConversations, getCustomerMood, sendMessage } from './services/api'
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
    customerName: `Hóspede ${shortId}`,
    initials: shortId.slice(0, 2),
    room: '',
    score: null,
    scale: '-1 to 1',
    lastSeen: '',
    messages: [],
  }
}

// Guest name/initials are never persisted server-side, only derived client-side from the id.
function toConversation(row: ConversationSummaryDto): Conversation {
  const shortId = row.customer_id.replace(/^customer-/, '').slice(0, 4).toUpperCase()
  const lastMessage = row.messages.at(-1)
  return {
    id: row.conversation_id,
    customerId: row.customer_id,
    customerName: `Hóspede ${shortId}`,
    initials: shortId.slice(0, 2),
    room: '',
    score: row.score,
    scale: row.scale ?? '-1 to 1',
    lastSeen: lastMessage ? formatTime(lastMessage.sent_at) : '',
    messages: row.messages.map((message) => ({
      id: message.message_id, role: message.role, text: message.text, time: formatTime(message.sent_at),
    })),
  }
}

function App() {
  const [conversations, setConversations] = useState<Conversation[]>([])
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [draft, setDraft] = useState('')
  const [sendRole, setSendRole] = useState<Role>('agent')
  const selectedConversation = conversations.find(({ id }) => id === selectedId) ?? null
  const messageAreaRef = useRef<HTMLDivElement>(null)

  // Hydrate from DuckDB-backed mood-api on load so reloads don't lose history.
  useEffect(() => {
    let cancelled = false
    getConversations()
      .then((rows) => {
        if (cancelled) return
        setConversations(rows.map(toConversation))
      })
      .catch(() => {
        // Best-effort hydration: an empty inbox is a safe fallback if mood-api is unreachable.
      })
    return () => {
      cancelled = true
    }
  }, [])

  // Jump to the newest message whenever the thread grows or the selection changes.
  useEffect(() => {
    const el = messageAreaRef.current
    if (!el) return
    el.scrollTop = el.scrollHeight
  }, [selectedConversation?.messages.length, selectedId])

  function handleAddConversation() {
    const conversation = createConversation()
    setConversations((current) => [conversation, ...current])
    setSelectedId(conversation.id)
  }

  async function handleDeleteConversation(conversationId: string) {
    setConversations((current) => current.filter((conversation) => conversation.id !== conversationId))
    setSelectedId((current) => (current === conversationId ? null : current))
    try {
      await deleteConversation(conversationId)
    } catch {
      // The conversation was already removed from the UI; a failed backend
      // delete just means it may reappear after the next reload.
    }
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
      return
    }
    if (event.key === 'Tab') {
      event.preventDefault()
      setSendRole((current) => (current === 'agent' ? 'customer' : 'agent'))
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
        <div className="brand-row"><div className="brand-mark">Unifor</div><h1>Central de Hóspedes</h1></div>
        <div className="inbox-heading"><div><p className="eyebrow">Suas reservas</p><h2>Hóspedes <span>{conversations.length}</span></h2></div><button className="compose-button" type="button" aria-label="Iniciar nova conversa" onClick={handleAddConversation}>+</button></div>
        <label className="search-box"><span aria-hidden="true">⌕</span><input type="search" placeholder="Buscar hóspedes" aria-label="Buscar hóspedes" /><kbd>/</kbd></label>
        <nav className="conversation-list" aria-label="Conversas">
          {conversations.length === 0 && <p className="empty-hint">Nenhum hóspede ainda. Clique em + para adicionar.</p>}
          {conversations.map((conversation) => (
            <div className={`conversation-item ${conversation.id === selectedId ? 'is-selected' : ''}`} key={conversation.id}>
              <button className="conversation-item-select" type="button" onClick={() => setSelectedId(conversation.id)}>
                <span className="avatar">{conversation.initials}</span>
                <span className="conversation-copy">
                  <span className="conversation-name">{getMoodEmoji(conversation.score, conversation.scale)} {conversation.customerName}</span>
                  <span className="conversation-preview">{conversation.messages.at(-1)?.text ?? 'Nenhuma mensagem ainda'}</span>
                </span>
                <span className="conversation-time">{conversation.lastSeen}</span>
              </button>
              <button className="icon-button conversation-delete" type="button" aria-label={`Excluir conversa de ${conversation.customerName}`} onClick={() => handleDeleteConversation(conversation.id)}>×</button>
            </div>
          ))}
        </nav>
      </aside>
      <section className="chat-panel" aria-label="Conversa selecionada">
        {selectedConversation ? (
          <>
            <header className="chat-header">
              <div className="guest-heading">
                <div className="avatar avatar-large">{selectedConversation.initials}</div>
                <div>
                  <div className="name-line"><h2>{selectedConversation.customerName}</h2><span className="online-dot" /><span className="online-label">Online</span></div>
                  <p>{selectedConversation.room || 'Nenhum quarto atribuído ainda'}</p>
                </div>
              </div>
              <div className="header-actions"><button className="secondary-button" type="button" onClick={refreshMood}><span>↻</span> Atualizar humor</button></div>
            </header>
            <div className="mood-strip">
              <div className="mood-icon">{getMoodEmoji(selectedConversation.score, selectedConversation.scale)}</div>
              <div><span className="eyebrow">Humor atual do hóspede</span><strong>{getMoodLabel(selectedConversation.score, selectedConversation.scale)}</strong></div>
              <div className="mood-meter" aria-label={`Pontuação de humor ${selectedConversation.score ?? 'indisponível'}`}><span style={{ width: selectedConversation.score === null ? '0%' : `${((selectedConversation.score + 1) / 2) * 100}%` }} /></div>
              <span className="mood-score">{selectedConversation.score === null ? '—' : `${selectedConversation.score > 0 ? '+' : ''}${selectedConversation.score.toFixed(2)}`}</span>
              <span className="model-tag">ML v0.1</span>
            </div>
            <div className="message-area" ref={messageAreaRef}>
              {selectedConversation.messages.length === 0 ? (
                <p className="empty-hint">Nenhuma mensagem ainda. Diga oi para começar o atendimento.</p>
              ) : (
                <>
                  <div className="date-divider"><span>Hoje</span></div>
                  <div className="messages">
                    {selectedConversation.messages.map((message) => (
                      <div className={`message-row ${message.role}`} key={message.id}>
                        {message.role === 'customer' && <span className="avatar avatar-small">{selectedConversation.initials}</span>}
                        <div className="message-content">
                          <div className="message-bubble">{message.text}</div>
                          <span className="message-time">{message.time}{message.role === 'agent' && '  ·  Lida'}</span>
                        </div>
                      </div>
                    ))}
                  </div>
                </>
              )}
            </div>
            <div className="composer-role-toggle" role="radiogroup" aria-label="Enviar mensagem como">
              <button type="button" className={sendRole === 'customer' ? 'is-active' : ''} onClick={() => setSendRole('customer')} aria-pressed={sendRole === 'customer'}>Simular hóspede</button>
              <button type="button" className={sendRole === 'agent' ? 'is-active' : ''} onClick={() => setSendRole('agent')} aria-pressed={sendRole === 'agent'}>Resposta da recepção</button>
            </div>
            <form className="composer" onSubmit={handleSend}>
              <textarea value={draft} onChange={(event) => setDraft(event.target.value)} onKeyDown={handleComposerKeyDown} placeholder="Escreva uma resposta..." aria-label="Mensagem" rows={1} />
              <div className="composer-actions"><span className="composer-hint">Enter para enviar · Shift+Enter para nova linha · Tab para trocar de papel</span><button className="send-button" type="submit" disabled={!draft.trim()} aria-label="Enviar mensagem">↑</button></div>
            </form>
          </>
        ) : (
          <div className="empty-panel"><p>Selecione uma conversa ou clique em + para adicionar um novo hóspede.</p></div>
        )}
      </section>
    </main>
  )
}

export default App

