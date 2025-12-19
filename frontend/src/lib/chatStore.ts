export interface StoredMessage {
    id: string;
    role: 'user' | 'assistant';
    content: string;
    timestamp: string; // ISO string for storage
}

export interface StoredChatMeta {
    id: string;
    title: string;
    createdAt: string;
    updatedAt: string;
    preview?: string;
}

const CHATS_KEY = 'noir_whisper_chats';
const MESSAGES_KEY_PREFIX = 'noir_whisper_messages_';
const STORE_EVENT = 'chat-store-changed';

function emitChange() {
    window.dispatchEvent(new Event(STORE_EVENT));
}

export function subscribe(listener: () => void) {
    window.addEventListener(STORE_EVENT, listener);
    return () => window.removeEventListener(STORE_EVENT, listener);
}

export function getChats(): StoredChatMeta[] {
    try {
        const raw = localStorage.getItem(CHATS_KEY);
        return raw ? JSON.parse(raw) as StoredChatMeta[] : [];
    } catch {
        return [];
    }
}

export function setChats(chats: StoredChatMeta[]) {
    localStorage.setItem(CHATS_KEY, JSON.stringify(chats));
    emitChange();
}

export function getMessages(chatId: string): StoredMessage[] {
    try {
        const raw = localStorage.getItem(MESSAGES_KEY_PREFIX + chatId);
        return raw ? JSON.parse(raw) as StoredMessage[] : [];
    } catch {
        return [];
    }
}

export function setMessages(chatId: string, messages: StoredMessage[]) {
    localStorage.setItem(MESSAGES_KEY_PREFIX + chatId, JSON.stringify(messages));
    // update preview and updatedAt on chat meta
    const chats = getChats();
    const chat = chats.find(c => c.id === chatId);
    if (chat) {
        chat.updatedAt = new Date().toISOString();
        chat.preview = messages.length ? messages[messages.length - 1].content.slice(0, 80) : '';
        setChats([...chats]);
    } else {
        emitChange();
    }
}

export function createChat(id: string) {
    const now = new Date().toISOString();
    const chats = getChats();
    const exists = chats.some(c => c.id === id);
    if (!exists) {
        chats.unshift({ id, title: 'New chat', createdAt: now, updatedAt: now, preview: '' });
        setChats(chats);
    }
    localStorage.setItem(MESSAGES_KEY_PREFIX + id, JSON.stringify([]));
}

export function deleteChat(id: string) {
    const chats = getChats().filter(c => c.id !== id);
    setChats(chats);
    localStorage.removeItem(MESSAGES_KEY_PREFIX + id);
}

export function setChatTitle(id: string, title: string) {
    const chats = getChats();
    const chat = chats.find(c => c.id === id);
    if (chat) {
        chat.title = title;
        chat.updatedAt = new Date().toISOString();
        setChats([...chats]);
    }
}


