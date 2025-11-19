document.addEventListener("DOMContentLoaded", () => {
    const feedbackForm = document.getElementById('feedbackForm');
    feedbackForm.addEventListener('submit', (e) => {
        e.preventDefault();
        const formData = new FormData(feedbackForm);
        fetch('/submit-feedback', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                feedbackForm.classList.add('error');
                alert(data.error);
                setTimeout(() => feedbackForm.classList.remove('error'), 500);
            } else {
                alert(data.message);
                feedbackForm.reset();
                window.location.reload(); // Refresh to show new feedback
            }
        })
        .catch(error => {
            feedbackForm.classList.add('error');
            alert('Error submitting feedback: ' + error.message);
            setTimeout(() => feedbackForm.classList.remove('error'), 500);
        });
    });
});