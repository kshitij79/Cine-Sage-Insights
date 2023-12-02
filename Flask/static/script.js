
document.addEventListener('DOMContentLoaded', (event) => {
  // Your entire JS code should be here
  // Define D3.js chart here
  const margin = { top: 20, right: 30, bottom: 40, left: 50 };
  const width = 1100 - margin.left - margin.right;
  const height = 600 - margin.top - margin.bottom;

  const svg = d3.select('#revenueChart')
    .append('svg')
    .attr('width', width + margin.left + margin.right)
    .attr('height', height + margin.top + margin.bottom)
    .append('g')
    .attr('transform', `translate(${margin.left},${margin.top})`);

  function generateChoroplethMap(revenueData) {

    var projection = d3.geoNaturalEarth1().scale(200)
    var path = d3.geoPath(projection);
    
    var revenueMap = new Map(revenueData.map(d => [getCountryCode(d.country), d.revenue]));
    // Load external data and boot
    var colorScale = d3.scaleThreshold()
        .domain([100000, 1000000, 10000000, 30000000, 100000000, 500000000])
        .range(d3.schemeReds[4]);
       
    d3.selectAll(".tooltip").remove();
    d3.json("https://raw.githubusercontent.com/holtzy/D3-graph-gallery/master/DATA/world.geojson")
        .then(world => {
            // Draw the map
            svg.append("g")
                .selectAll("path")
                .data(world.features)
                .enter()
                .append("path")
                .attr("d", path)
                .attr("fill", function (d) {
                    // Use revenue data for coloring
                    var revenue = revenueMap.get(d.id) || 0;
                    if (revenue != 0) {
                      var tooltip = d3.select("body").append("div")
                      .attr("class", "tooltip tooltip-" + d.id) // Unique class for each tooltip
                      .style("opacity", 0);
                      var currentObjectAbsolutePosition = this.getBoundingClientRect();
                      var countryX = currentObjectAbsolutePosition.x;

                      var countryAbsolutePathLocation = currentObjectAbsolutePosition.y + window.scrollY;
                      countryX = countryX;
                      countryY = countryAbsolutePathLocation + 20;

                      tooltip.transition()
                      .duration(200)
                      .style("opacity", .8);
                      tooltip.html(`Country: ${d.properties.name}<br>Revenue: ${revenue}`)
                      .style("left", (countryX) + "px")
                      .style("top", (countryY) + "px");
                    }
                    return colorScale(revenue);
                }).style("pointer-events", "all").on("mouseover", function(event, d) {
                    console.log(d);
                })
                .on("mouseout", function(d) {
                    console.log(d);
                });
        })
   
  }

  document.getElementById('predict').addEventListener('click', function () {

    console.log('Button clicked');
    // Fetch input values
    const prompt = document.getElementById('prompt').value;
    const budget = document.getElementById('budget').value;
    // const country = document.getElementById('country').value;
    const language = document.getElementById('language').value;

    // Make an API request to your Flask server
    fetch('/predict', {
      method: 'POST',
      body: JSON.stringify({ prompt, budget, language }),
      headers: {
        'Content-Type': 'application/json',
      },
    })
      .then((response) => response.json())
      .then((data) => {
        // Handle the API response here
        // You can update the chart or display the result as needed
        console.log('API Response:', data);
        generateChoroplethMap(data.revenueData);
      })
      .catch((error) => {
        console.error(error);
      });
  });

  function getCountryCode(countryName) {
    // Mapping from country names to three-letter country codes
    const countryCodeMap = {
        'USA': 'USA', 'UK': 'GBR', 'UAE': 'ARE', 'India': 'IND', 
        'Afghanistan': 'AFG', 'Albania': 'ALB', 'Algeria': 'DZA', 
        'Andorra': 'AND', 'Angola': 'AGO', 'Antigua and Barbuda': 'ATG', 
        'Argentina': 'ARG', 'Armenia': 'ARM', 'Australia': 'AUS', 
        'Austria': 'AUT', 'Azerbaijan': 'AZE', 'Bahamas': 'BHS', 
        'Bahrain': 'BHR', 'Bangladesh': 'BGD', 'Barbados': 'BRB', 
        'Belarus': 'BLR', 'Belgium': 'BEL', 'Belize': 'BLZ', 
        'Benin': 'BEN', 'Bhutan': 'BTN', 'Bolivia': 'BOL', 
        'Bosnia and Herzegovina': 'BIH', 'Botswana': 'BWA', 'Brazil': 'BRA', 
        'Brunei': 'BRN', 'Bulgaria': 'BGR', 'Burkina Faso': 'BFA', 
        'Burundi': 'BDI', 'Cabo Verde': 'CPV', 'Cambodia': 'KHM', 
        'Cameroon': 'CMR', 'Canada': 'CAN', 'Central African Republic': 'CAF', 
        'Chad': 'TCD', 'Chile': 'CHL', 'China': 'CHN', 
        'Colombia': 'COL', 'Comoros': 'COM', 'Congo, Democratic Republic of the': 'COD', 
        'Congo, Republic of the': 'COG', 'Costa Rica': 'CRI', 'Ivoire': 'CIV', 
        'Croatia': 'HRV', 'Cuba': 'CUB', 'Cyprus': 'CYP', 
        'Czech Republic': 'CZE', 'Denmark': 'DNK', 'Djibouti': 'DJI', 
        'Dominica': 'DMA', 'Dominican Republic': 'DOM', 'East Timor': 'TLS', 
        'Ecuador': 'ECU', 'Egypt': 'EGY', 'El Salvador': 'SLV', 
        'Equatorial Guinea': 'GNQ', 'Eritrea': 'ERI', 'Estonia': 'EST', 
        'Eswatini': 'SWZ', 'Ethiopia': 'ETH', 'Fiji': 'FJI', 
        'Finland': 'FIN', 'France': 'FRA', 'Gabon': 'GAB', 
        'Gambia': 'GMB', 'Georgia': 'GEO', 'Germany': 'DEU', 
        'Ghana': 'GHA', 'Greece': 'GRC', 'Grenada': 'GRD', 
        'Guatemala': 'GTM', 'Guinea': 'GIN', 'Guinea-Bissau': 'GNB', 
        'Guyana': 'GUY', 'Haiti': 'HTI', 'Honduras': 'HND', 
        'Hungary': 'HUN', 'Iceland': 'ISL', 'Indonesia': 'IDN', 
        'Iran': 'IRN', 'Iraq': 'IRQ', 'Ireland': 'IRL', 
        'Israel': 'ISR', 'Italy': 'ITA', 'Jamaica': 'JAM', 
        'Japan': 'JPN', 'Jordan': 'JOR', 'Kazakhstan': 'KAZ', 
        'Kenya': 'KEN', 'Kiribati': 'KIR', 'Korea, North': 'PRK', 
        'Korea, South': 'KOR', 'Kosovo': 'XKX', 'Kuwait': 'KWT', 
        'Kyrgyzstan': 'KGZ', 'Laos': 'LAO', 'Latvia': 'LVA', 
        'Lebanon': 'LBN', 'Lesotho': 'LSO', 'Liberia': 'LBR', 
        'Libya': 'LBY', 'Liechtenstein': 'LIE', 'Lithuania': 'LTU', 
        'Luxembourg': 'LUX', 'Madagascar': 'MDG', 'Malawi': 'MWI', 
        'Malaysia': 'MYS', 'Maldives': 'MDV', 'Mali': 'MLI', 
        'Malta': 'MLT', 'Marshall Islands': 'MHL', 'Mauritania': 'MRT', 
        'Mauritius': 'MUS', 'Mexico': 'MEX', 'Micronesia': 'FSM', 
        'Moldova': 'MDA', 'Monaco': 'MCO', 'Mongolia': 'MNG', 
        'Montenegro': 'MNE', 'Morocco': 'MAR', 'Mozambique': 'MOZ', 
        'Myanmar': 'MMR', 'Namibia': 'NAM', 'Nauru': 'NRU', 
        'Nepal': 'NPL', 'Netherlands': 'NLD', 'New Zealand': 'NZL', 
        'Nicaragua': 'NIC', 'Niger': 'NER', 'Nigeria': 'NGA', 
        'North Macedonia': 'MKD', 'Norway': 'NOR', 'Oman': 'OMN', 
        'Pakistan': 'PAK', 'Palau': 'PLW', 'Palestine': 'PSE', 
        'Panama': 'PAN', 'Papua New Guinea': 'PNG', 'Paraguay': 'PRY', 
        'Peru': 'PER', 'Philippines': 'PHL', 'Poland': 'POL', 
        'Portugal': 'PRT', 'Qatar': 'QAT', 'Romania': 'ROU', 
        'Russia': 'RUS', 'Rwanda': 'RWA', 'Saint Kitts and Nevis': 'KNA', 
        'Saint Lucia': 'LCA', 'Saint Vincent and the Grenadines': 'VCT', 
        'Samoa': 'WSM', 'San Marino': 'SMR', 'Sao Tome and Principe': 'STP', 
        'Saudi Arabia': 'SAU', 'Senegal': 'SEN', 'Serbia': 'SRB', 
        'Seychelles': 'SYC', 'Sierra Leone': 'SLE', 'Singapore': 'SGP', 
        'Slovakia': 'SVK', 'Slovenia': 'SVN', 'Solomon Islands': 'SLB', 
        'Somalia': 'SOM', 'South Africa': 'ZAF', 'South Sudan': 'SSD', 
        'Spain': 'ESP', 'Sri Lanka': 'LKA', 'Sudan': 'SDN', 
        'Suriname': 'SUR', 'Sweden': 'SWE', 'Switzerland': 'CHE', 
        'Syria': 'SYR', 'Taiwan': 'TWN', 'Tajikistan': 'TJK', 
        'Tanzania': 'TZA', 'Thailand': 'THA', 'Togo': 'TGO', 
        'Tonga': 'TON', 'Trinidad and Tobago': 'TTO', 'Tunisia': 'TUN', 
        'Turkey': 'TUR', 'Turkmenistan': 'TKM', 'Tuvalu': 'TUV', 
        'Uganda': 'UGA', 'Ukraine': 'UKR', 'Uruguay': 'URY', 
        'Uzbekistan': 'UZB', 'Vanuatu': 'VUT', 'Vatican City': 'VAT', 
        'Venezuela': 'VEN', 'Vietnam': 'VNM', 'Yemen': 'YEM', 
        'Zambia': 'ZMB', 'Zimbabwe': 'ZWE'
    };

    // Return the three-letter country code from the map
    return countryCodeMap[countryName] || ''; 
}

});
