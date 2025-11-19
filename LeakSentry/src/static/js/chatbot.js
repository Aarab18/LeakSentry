document.addEventListener("DOMContentLoaded", () => {
    const chatbotToggle = document.getElementById('chatbotToggle');
    const chatbotWindow = document.getElementById('chatbotWindow');
    const chatbotBody = document.getElementById('chatbotBody');
    const chatbotInput = document.getElementById('chatbotInput');
    const chatbotSend = document.getElementById('chatbotSend');

    chatbotToggle.addEventListener('click', () => {
        chatbotWindow.classList.toggle('open');
    });

    chatbotSend.addEventListener('click', sendMessage);
    chatbotInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') sendMessage();
    });

    function sendMessage() {
        const message = chatbotInput.value.trim();
        if (!message) return;

        const userMessage = document.createElement('div');
        userMessage.className = 'chatbot-message user';
        userMessage.textContent = message;
        chatbotBody.appendChild(userMessage);

        chatbotInput.value = '';

        fetch('/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message })
        })
        .then(response => response.json())
        .then(data => {
            const botMessage = document.createElement('div');
            botMessage.className = 'chatbot-message bot';
            botMessage.textContent = data.response || 'Sorry, I encountered an error.';
            chatbotBody.appendChild(botMessage);
            chatbotBody.scrollTop = chatbotBody.scrollHeight;
        })
        .catch(error => {
            const botMessage = document.createElement('div');
            botMessage.className = 'chatbot-message bot';
            botMessage.textContent = 'Error: Could not reach the server.';
            chatbotBody.appendChild(botMessage);
            chatbotBody.scrollTop = chatbotBody.scrollHeight;
        });
    }
});