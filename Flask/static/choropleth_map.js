
function choropleth_map()
{
    var map_container = document.createElement('div');

    var paragraph = document.createElement('p');
    // Set the text content of the <p> element
    paragraph.textContent = 'Additional information in the paragraph.';

    // Append the text node and the <p> element to the new div
    map_container.appendChild(paragraph);
     // define the dimensions and margins for the graph
     margin = ({top: 100, right: 100, bottom: 100, left: 100})
     width = 960
     height = 500
 
     // define function to parse time in years format
 
     // create scales x & y for X and Y axis and set their ranges
     x = d3.scaleTime()
           .range([margin.left, width - margin.right]);
     y = d3.scaleLinear()
           .range([height - margin.bottom, margin.top]);
 
 
     // append svg element to the body of the page
     // set dimensions and position of the svg element
     var svg = d3.select("body")
               .append("svg")
               .attr("id", "svg1")
               .attr("width", width)
               .attr("height", height);
     
     var container = svg.append("g")
             .attr("id", "container")
    return svg
    
}
