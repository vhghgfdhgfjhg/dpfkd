document.getElementById('upload-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    const videoFile = document.getElementById('video-upload').files[0];
    const formData = new FormData();
    formData.append('video', videoFile);

    try {
        const response = await fetch('/detect', {
            method: 'POST',
            body: formData
        });
        const result = await response.json();
        
        document.getElementById('result-container').style.display = 'block';
        document.getElementById('accuracy-result').innerText = 
            Accuracy: ${result.accuracy}%;
        document.getElementById('fake-real-status').innerText = 
            result.is_fake ? 'Detected as FAKE' : 'Detected as REAL';
    } catch (error) {
        console.error('Error:', error);
    }
});