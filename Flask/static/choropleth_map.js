
function choropleth_map()
{
    console.log("inside the function");
    var map_container = document.createElement('div');

    var paragraph = document.createElement('p');
    // Set the text content of the <p> element
    paragraph.textContent = 'Additional information in the paragraph.';

    // Append the text node and the <p> element to the new div
    map_container.appendChild(paragraph);

    return map_container
    
}
