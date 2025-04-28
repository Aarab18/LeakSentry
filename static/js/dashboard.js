document.addEventListener('DOMContentLoaded', () => {
    const ctx = document.getElementById('severityChart').getContext('2d');
    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['Critical', 'Moderate', 'Low'],
            datasets: [{
                label: 'Leak Severity Count',
                data: [
                    severityCounts.Critical,
                    severityCounts.Moderate,
                    severityCounts.Low
                ],
                backgroundColor: [
                    'rgba(255, 99, 132, 0.6)',  // Critical: Red
                    'rgba(255, 159, 64, 0.6)',  // Moderate: Orange
                    'rgba(75, 192, 192, 0.6)'   // Low: Teal
                ],
                borderColor: [
                    'rgba(255, 99, 132, 1)',
                    'rgba(255, 159, 64, 1)',
                    'rgba(75, 192, 192, 1)'
                ],
                borderWidth: 1
            }]
        },
        options: {
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Number of Leaks'
                    }
                },
                x: {
                    title: {
                        display: true,
                        text: 'Severity'
                    }
                }
            },
            plugins: {
                legend: {
                    display: false
                }
            }
        }
    });
});