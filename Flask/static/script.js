
document.addEventListener('DOMContentLoaded', (event) => {
  // Your entire JS code should be here
  // Define D3.js chart here
  const margin = { top: 20, right: 30, bottom: 40, left: 50 };
  const width = 600 - margin.left - margin.right;
  const height = 400 - margin.top - margin.bottom;

  const svg = d3.select('#revenueChart')
    .append('svg')
    .attr('width', width + margin.left + margin.right)
    .attr('height', height + margin.top + margin.bottom)
    .append('g')
    .attr('transform', `translate(${margin.left},${margin.top})`);

  // Function to update the D3.js chart
  function updateChart(data) {
    // Clear the previous chart content
    svg.selectAll('*').remove();

    // Implement your D3.js chart logic here using 'data'
    // Example: Create a bar chart
    const x = d3.scaleBand()
      .domain(data.map(d => d.country))
      .range([0, width])
      .padding(0.1);

    const y = d3.scaleLinear()
      .domain([0, d3.max(data, d => d.revenue)])
      .nice()
      .range([height, 0]);

    svg.append('g')
      .selectAll('rect')
      .data(data)
      .enter()
      .append('rect')
      .attr('x', d => x(d.country))
      .attr('y', d => y(d.revenue))
      .attr('width', x.bandwidth())
      .attr('height', d => height - y(d.revenue))
      .attr('fill', 'steelblue');

    svg.append('g')
      .attr('class', 'x-axis')
      .attr('transform', `translate(0,${height})`)
      .call(d3.axisBottom(x));

    svg.append('g')
      .attr('class', 'y-axis')
      .call(d3.axisLeft(y));

    // Add labels, titles, or any other necessary chart elements here
  }

  document.getElementById('predict').addEventListener('click', function () {
    // Fetch input values
    const prompt = document.getElementById('prompt').value;
    const budget = document.getElementById('budget').value;
    const country = document.getElementById('country').value;
    const language = document.getElementById('language').value;

    // Make an API request to your Python script
    fetch('/predict', {
      method: 'POST',
      body: JSON.stringify({ prompt, budget, country, language }),
      headers: {
        'Content-Type': 'application/json',
      },
    })
      .then((response) => response.json())
      .then((data) => {
        // Update the D3.js chart with the data
        updateChart(data.revenueData); // Assuming 'revenueData' is the JSON key returned by your Python script
      })
      .catch((error) => {
        console.error(error);
      });
  });

  document.getElementById('predict').addEventListener('click', function () {

    console.log('Button clicked');
    // Fetch input values
    const prompt = document.getElementById('prompt').value;
    const budget = document.getElementById('budget').value;
    const country = document.getElementById('country').value;
    const language = document.getElementById('language').value;

    // Make an API request to your Flask server
    fetch('/predict', {
      method: 'POST',
      body: JSON.stringify({ prompt, budget, country, language }),
      headers: {
        'Content-Type': 'application/json',
      },
    })
      .then((response) => response.json())
      .then((data) => {
        // Handle the API response here
        // You can update the chart or display the result as needed
        console.log('API Response:', data);
      })
      .catch((error) => {
        console.error(error);
      });
  });


});
