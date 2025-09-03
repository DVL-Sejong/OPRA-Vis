var heatWidth = document.getElementById('heatmap').clientWidth;
var heatHeight = document.getElementById('heatmap').clientHeight - margin.top - margin.bottom;

var selectedClass = undefined;
var prevSelectedClass = undefined;

var colorMaps = {};

function resetSelections() {
    d3.selectAll(".scatter").style("fill", function() {
        var cls = d3.select(this).attr("class").split(" ").find(cl => cl.startsWith("data-point-")).slice(11);
        return colorMaps[cls];
    });
    d3.selectAll(".dot").style("fill", function() { return this.classList.contains("sentiment-1")? '#4472C4': 'red'; });
    d3.selectAll(".line").style("stroke", "lightgrey").style("opacity", 0.5).style('stroke-width', '1px');

    selectedDots = [];
    lassoPath.style("display", "none");
}

function highlightSelection(selectedClass) {
    d3.selectAll(".scatter-" + selectedClass).style("fill", "yellow");
    d3.selectAll(".line-" + selectedClass).style("stroke", "yellow").style("opacity", 1).style("stroke-width", "3px");
    d3.selectAll(".dot-" + selectedClass).style("fill", "yellow");
}

const heatSvg = d3.select('#heatmap').append("svg")
        .attr("width", heatWidth + margin.left + margin.right)
        .attr("height", heatHeight + margin.top + margin.bottom)
    .append("g")
        .attr("transform",
            `translate(0,${margin.top})`);

d3.csv("static/data/positive_sentiment.csv").then( function(data) {
    let newData = data.map(function(d) {
        return {
            idx: d[Object.keys(d)[0]],
            sentence: d.sentence,
            sentiment: d.sentiment
        }
    });

    heatSvg.selectAll('circle')
        .data(newData)
        .enter().append('circle')
        .attr('class', d => 'dot dot-data-point-' + d.idx + ' sentiment-' + d.sentiment + ' data-point-' + d.idx)
        .attr('cx', d => d.sentiment == 1? Math.random() * (heatWidth/2-10) + 5 : Math.random() * (heatWidth/2-10) + heatWidth/2 + 5)
        .attr('cy', d => Math.random() * heatHeight + 5)
        .attr('r', 2)
        .style('fill', d => d.sentiment == 1 ? '#4472C4' : 'red')
        .on('click', function(event, d) {
            resetSelections();

                selectedClass = d3.select(this).attr("class").split(" ").find(cl => cl.startsWith("data-point-"));

                if(selectedClass == prevSelectedClass) {
                    resetSelections();
                    selectedClass = undefined;
                    prevSelectedClass = undefined;
                }

                else {
                    highlightSelection(selectedClass);
                    prevSelectedClass = selectedClass;
                }
        });

    heatSvg.append('line')
        .attr('x1', heatWidth / 2)
        .attr('y1', 0)
        .attr('x2', heatWidth / 2)
        .attr('y2', heatHeight + margin.top)
        .style('stroke', 'green')
        .style('stroke-width', 2);

    heatSvg.append('line')
        .attr('x1', 0)
        .attr('y1', 0)
        .attr('x2', heatWidth)
        .attr('y2', 0)
        .style('stroke', 'black')
        .style('stroke-width', 1);

    heatSvg.append('text')
        .attr('x', heatWidth/4)
        .attr('y', 0 - margin.top/2)
        .style('font-size', '14px')
        .style('fill', 'black')
        .style('text-anchor', 'middle')
        .text('Positive')

    heatSvg.append('text')
        .attr('x', heatWidth/4 * 3)
        .attr('y', 0 - margin.top/2)
        .style('font-size', '14px')
        .style('fill', 'black')
        .style('text-anchor', 'middle')
        .text('Negative')
});