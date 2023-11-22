function choropleth_map()
{
    alert("Choropleth map is ready!");
    var svg  =d3.select("body")
                .append("svg")
                .attr("id", "”choropleth”")
                .attr("width", 700)
                .attr("height", 700)


    // Map and projection
    
    var projection = d3.geoNaturalEarth1().scale(200)
    var path = d3.geoPath(projection);

    // Data and color scale
    var data = d3.map();
    var colorScale = d3.scaleThreshold()
    .domain([100000, 1000000, 10000000, 30000000, 100000000, 500000000])
    .range(d3.schemeBlues[7]);

    // Load external data and boot
    d3.queue()
    .defer(d3.json, "https://raw.githubusercontent.com/holtzy/D3-graph-gallery/master/DATA/world.geojson")
    .defer(d3.csv, "https://raw.githubusercontent.com/holtzy/D3-graph-gallery/master/DATA/world_population.csv", function(d) { data.set(d.code, +d.pop); })
    .await(ready);

    function ready(error, topo) 
    {

        // Draw the map
        svg.append("g")
            .selectAll("path")
            .data(topo.features)
            .enter()
            .append("path")
            // draw each country
            .attr("d", path
            )
            // set the color of each country
            .attr("fill", function (d) {
                d.total = data.get(d.id) || 0;
                return colorScale(d.total);
            });
    }
    return svg.node();
}