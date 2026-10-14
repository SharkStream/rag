import { useEffect, useMemo, useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import './App.css'

const emptyDocumentState = 'No documents indexed yet.'

function App() {
  const [messages, setMessages] = useState([])
  const [documents, setDocuments] = useState([])
  const [question, setQuestion] = useState('')
  const [status, setStatus] = useState('Ready to index documents.')
  const [loading, setLoading] = useState(false)
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false)
  const [fileInput, setFileInput] = useState(null)
  const [folderPath, setFolderPath] = useState('')
  const [prompts, setPrompts] = useState([])
  const [selectedPrompt, setSelectedPrompt] = useState('rag.answer_with_context')

  const formattedDocuments = useMemo(
    () => documents.filter(Boolean),
    [documents],
  )

  const selectedPromptMeta = useMemo(
    () => prompts.find((prompt) => prompt.name === selectedPrompt) ?? null,
    [prompts, selectedPrompt],
  )

  const fetchDocuments = async (source = null) => {
    try {
      const query = source ? `?source=${encodeURIComponent(source)}` : ''
      const response = await fetch(`/api/documents${query}`)
      const data = await response.json()
      setDocuments(Array.isArray(data.documents) ? data.documents : [])
    } catch (error) {
      setDocuments([])
      setStatus('Unable to load document list.')
    }
  }

  useEffect(() => {
    fetchDocuments()
  }, [])

  useEffect(() => {
    const fetchPrompts = async () => {
      try {
        const response = await fetch('/api/prompts')
        const data = await response.json()
        const promptList = Array.isArray(data.prompts) ? data.prompts : []
        setPrompts(promptList)

        if (promptList.length > 0) {
          const preferredPrompt =
            promptList.find((prompt) => prompt.name === 'rag.answer_with_context') ??
            promptList.find((prompt) => prompt.name.startsWith('rag.')) ??
            promptList[0]
          setSelectedPrompt(preferredPrompt.name)
        }
      } catch (error) {
        setPrompts([])
      }
    }

    fetchPrompts()
  }, [])

  const deleteDocument = async (sourceName) => {
    try {
      const response = await fetch('/api/documents/delete', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ source: sourceName }),
      })
      const data = await response.json()

      if (!response.ok) {
        throw new Error(data.error || 'Delete document failed')
      }

      setStatus(`Deleted source: ${sourceName}`)
      await fetchDocuments()
    } catch (error) {
      setStatus(error.message || 'Delete document failed.')
    }
  }

  const handleUpload = async () => {
    if (!fileInput || fileInput.files.length === 0) {
      setStatus('Please choose at least one file to upload.', true)
      return
    }

    const formData = new FormData()
    Array.from(fileInput.files).forEach((file) => formData.append('file', file))

    setLoading(true)
    setStatus('Uploading and indexing files...')

    try {
      const response = await fetch('/api/upload', {
        method: 'POST',
        body: formData,
      })
      const data = await response.json()

      if (!response.ok) {
        throw new Error(data.error || 'Upload failed')
      }

      setStatus(data.message || 'Files uploaded successfully.')
      fileInput.value = ''
      await fetchDocuments()
    } catch (error) {
      setStatus(error.message || 'Upload failed.')
    } finally {
      setLoading(false)
    }
  }

  const handleFolderIndex = async () => {
    if (!folderPath.trim()) {
      setStatus('Please enter a valid folder path.', true)
      return
    }

    setLoading(true)
    setStatus('Indexing folder...')

    try {
      const response = await fetch('/api/upload-folder', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ folder_path: folderPath }),
      })
      const data = await response.json()

      if (!response.ok) {
        throw new Error(data.error || 'Folder indexing failed')
      }

      setStatus(`Indexed ${data.chunks} chunks from ${data.files.length} file(s).`)
      setFolderPath('')
      await fetchDocuments()
    } catch (error) {
      setStatus(error.message || 'Folder indexing failed.')
    } finally {
      setLoading(false)
    }
  }

  const handleClearCollection = async () => {
    try {
      const response = await fetch('/api/clear', { method: 'POST' })
      const data = await response.json()

      if (!response.ok) {
        throw new Error(data.error || 'Clear collection failed')
      }

      setStatus(data.message || 'Knowledge base cleared.')
      await fetchDocuments()
    } catch (error) {
      setStatus(error.message || 'Clear collection failed.')
    }
  }

  const handleSubmit = async (event) => {
    event.preventDefault()
    const trimmed = question.trim()
    if (!trimmed || loading) {
      return
    }

    const userMessage = { id: Date.now(), type: 'user', content: trimmed }
    const targetMessages = [...messages, userMessage]
    setMessages(targetMessages)
    setQuestion('')
    setLoading(true)
    setStatus('Searching the knowledge base...')

    try {
      const response = await fetch('/api/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          question: trimmed,
          prompt_name: selectedPrompt,
        }),
      })
      const data = await response.json()

      if (!response.ok) {
        throw new Error(data.error || 'Query failed')
      }

      const answer = data.answer || 'No answer returned.'
      const sources = Array.isArray(data.sources) ? data.sources : []
      setMessages((prev) => [
        ...prev,
        { id: Date.now() + 1, type: 'assistant', content: answer, sources, feedback: null },
      ])
      setStatus('Query finished.')
    } catch (error) {
      setMessages((prev) => [
        ...prev,
        { id: Date.now() + 2, type: 'assistant', content: `Error: ${error.message}`, feedback: null },
      ])
      setStatus(error.message || 'Query failed.')
    } finally {
      setLoading(false)
    }
  }

  const handleMessageAction = async (messageId, action, payload = {}) => {
    try {
      const response = await fetch('/api/chat/action', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action, message_id: messageId, ...payload }),
      })
      const data = await response.json()

      if (!response.ok) {
        throw new Error(data.error || 'Action failed')
      }

      return true
    } catch (error) {
      setStatus(error.message || 'Action failed.')
      return false
    }
  }

  const handleCopyMessage = async (message) => {
    try {
      if (!navigator.clipboard || !message?.content) {
        throw new Error('Unable to copy content.')
      }

      await navigator.clipboard.writeText(message.content)
      setStatus('Copied response to clipboard.')
    } catch (error) {
      setStatus(error.message || 'Copy failed.')
    }
  }

  const handleRegenerateMessage = async (messageId) => {
    const targetIndex = messages.findIndex((message) => message.id === messageId)
    if (targetIndex === -1) {
      setStatus('Unable to find the response to regenerate.')
      return
    }

    const previousUserMessage = [...messages]
      .slice(0, targetIndex)
      .reverse()
      .find((message) => message.type === 'user')

    if (!previousUserMessage?.content) {
      setStatus('No valid question found to regenerate.')
      return
    }

    setLoading(true)
    setStatus('Regenerating response...')

    try {
      const response = await fetch('/api/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          question: previousUserMessage.content,
          prompt_name: selectedPrompt,
        }),
      })
      const data = await response.json()

      if (!response.ok) {
        throw new Error(data.error || 'Regenerate failed')
      }

      const answer = data.answer || 'No answer returned.'
      const sources = Array.isArray(data.sources) ? data.sources : []

      setMessages((prev) => prev.map((message) => (
        message.id === messageId
          ? { ...message, content: answer, sources, feedback: null }
          : message
      )))
      setStatus('Regenerated response.')
    } catch (error) {
      setStatus(error.message || 'Regeneration failed.')
    } finally {
      setLoading(false)
    }
  }

  const handleFeedback = async (messageId, nextFeedback) => {
    const success = await handleMessageAction(messageId, nextFeedback === 'good' ? 'like' : 'dislike')
    if (!success) {
      return
    }

    setMessages((prev) => prev.map((message) => {
      if (message.id !== messageId || message.type !== 'assistant') {
        return message
      }

      const currentFeedback = message.feedback
      const updatedFeedback = currentFeedback === nextFeedback ? null : nextFeedback
      return { ...message, feedback: updatedFeedback }
    }))

    setStatus(nextFeedback === 'good' ? 'Response marked as helpful.' : 'Response marked as not helpful.')
  }

  return (
    <div className="app-shell">
      <aside className={`sidebar ${sidebarCollapsed ? 'collapsed' : ''}`}>
        <div className="sidebar-header">
          {!sidebarCollapsed && <h2>Documents</h2>}
          <button
            type="button"
            className="sidebar-toggle"
            onClick={() => setSidebarCollapsed((value) => !value)}
            aria-label={sidebarCollapsed ? 'Expand sidebar' : 'Collapse sidebar'}
          >
            {sidebarCollapsed ? '▷' : '▼'}
          </button>
        </div>

        <div className="sidebar-section">
          <div className="sidebar-section-header">
            <span className="section-icon">⬆️</span>
            <span>Upload</span>
          </div>

          <label className="upload-box">
            <span>Upload files</span>
            <input
              type="file"
              multiple
              onChange={(event) => setFileInput(event.target)}
            />
          </label>

          <button type="button" className="primary-btn" onClick={handleUpload}>
            Upload &amp; Index
          </button>
        </div>

        <div className="sidebar-section">
          <div className="sidebar-section-header">
            <span className="section-icon">✨</span>
            <span>Prompt</span>
          </div>

          <div className="prompt-box">
            <label htmlFor="prompt-select" className="prompt-label">
              Prompt
            </label>
            <select
              id="prompt-select"
              value={selectedPrompt}
              onChange={(event) => setSelectedPrompt(event.target.value)}
              disabled={prompts.length === 0}
            >
              {prompts.length === 0 ? (
                <option value="">Loading prompts...</option>
              ) : (
                prompts.map((prompt) => (
                  <option key={prompt.name} value={prompt.name}>
                    {prompt.name}
                  </option>
                ))
              )}
            </select>
            {selectedPromptMeta && (
              <span className="prompt-description">{selectedPromptMeta.description}</span>
            )}
          </div>
        </div>

        <div className="sidebar-section">
          <div className="sidebar-section-header">
            <span className="section-icon">📚</span>
            <span>Documents</span>
          </div>

          <div className="folder-box">
            <label htmlFor="folderPath">Folder path</label>
            <input
              id="folderPath"
              value={folderPath}
              onChange={(event) => setFolderPath(event.target.value)}
              placeholder="D:/docs or /home/user/docs"
            />
            <button type="button" className="secondary-btn" onClick={handleFolderIndex}>
              Index Folder
            </button>
          </div>

          <button type="button" className="danger-btn" onClick={handleClearCollection}>
            Clear Collection
          </button>

          <div className="status-box">
            <h3>Status</h3>
            <p>{status}</p>
          </div>

          <div className="status-box">
            <h3>Indexed sources</h3>
            <ul className="doc-list">
              {formattedDocuments.length === 0 ? (
                <li>{emptyDocumentState}</li>
              ) : (
                formattedDocuments.map((item) => {
                  const sourceName = item.payload?.source || item.id || 'Unnamed source'
                  return (
                    <li key={sourceName} className="doc-list-item">
                      <span className="doc-label">{sourceName}</span>
                      <button
                        type="button"
                        className="doc-delete-btn"
                        onClick={() => deleteDocument(sourceName)}
                      >
                        Delete
                      </button>
                    </li>
                  )
                })
              )}
            </ul>
          </div>
        </div>
      </aside>

      <main className="main-panel">
        <div className="messages-area">
          <div className="messages-content">
            {messages.length === 0 && (
              <div className="welcome-state">
                <div className="welcome-icon">✦</div>
                <h2>RAG Knowledge Assistant</h2>
                <p>
                  Upload your documents, index them into Qdrant, and ask questions
                  about the content.
                </p>
              </div>
            )}

            {messages.map((message) => (
              <div key={message.id} className={`message-row ${message.type}`}>
                <div className="message-card">
                  <div className="message-meta">
                    <strong>{message.type === 'user' ? 'You' : 'AI Assistant'}</strong>
                  </div>

                  {message.type === 'user' ? (
                    <div className="message-content user-content">{message.content}</div>
                  ) : (
                    <div className="message-content markdown-body">
                      <ReactMarkdown remarkPlugins={[remarkGfm]}>{message.content}</ReactMarkdown>
                    </div>
                  )}

                  {message.sources && message.sources.length > 0 && (
                    <div className="source-list">
                      {message.sources.map((source) => (
                        <div key={`${source.id ?? source.source ?? 'source'}-${source.score ?? 0}`} className="source-item">
                          <span>{source.source || 'unknown'}</span>
                          <span>score: {Number(source.score || 0).toFixed(4)}</span>
                        </div>
                      ))}
                    </div>
                  )}

                  {message.type === 'assistant' && (
                    <div className="message-feedback" aria-label="Response actions">
                      <button
                        type="button"
                        className="message-action-btn"
                        data-tooltip="Copy response"
                        onClick={() => handleCopyMessage(message)}
                        aria-label="Copy response"
                      >
                        ⧉
                      </button>

                      <button
                        type="button"
                        className="message-action-btn"
                        data-tooltip="Regenerate"
                        onClick={() => handleRegenerateMessage(message.id)}
                        aria-label="Regenerate response"
                      >
                        ↻
                      </button>

                      <button
                        type="button"
                        className={`message-action-btn ${message.feedback === 'good' ? 'active good' : ''}`}
                        data-tooltip="Good response"
                        onClick={() => handleFeedback(message.id, 'good')}
                        aria-label="Good response"
                        aria-pressed={message.feedback === 'good'}
                      >
                        👍
                      </button>

                      <button
                        type="button"
                        className={`message-action-btn ${message.feedback === 'bad' ? 'active bad' : ''}`}
                        data-tooltip="Bad response"
                        onClick={() => handleFeedback(message.id, 'bad')}
                        aria-label="Bad response"
                        aria-pressed={message.feedback === 'bad'}
                      >
                        👎
                      </button>
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="input-bar">
          <form onSubmit={handleSubmit} className="chat-form">
            <textarea
              value={question}
              rows={1}
              onChange={(event) => setQuestion(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === 'Enter' && !event.shiftKey) {
                  event.preventDefault();

                  if (question.trim() && !loading) {
                    handleSubmit(event);
                  }
                }
              }}
              placeholder="Ask a question about your indexed documents..."
            />
            <button type="submit" className="send-btn" disabled={!question.trim() || loading}>
              {loading ? '...' : 'Send'}
            </button>
          </form>
        </div>
      </main>
    </div>
  )
}

export default App
