const margin = {top: 30, right: 30, bottom: 30, left: 30};
var width = document.getElementById('parallel').clientWidth - margin.left - margin.right;
var height = document.getElementById('parallel').clientHeight - margin.top - margin.bottom;

const svg = d3.select("#parallel")
    .append("svg")
        .attr("width", width + margin.left + margin.right)
        .attr("height", height + margin.top + margin.bottom)
    .append("g")
        .attr("transform",
            `translate(${margin.left},${margin.top})`);

d3.csv("static/data/unique_column.csv").then( function(data) {
    const color = "lightgrey"

    let newData = data.map(function(d) {
        return {
            idx: d[Object.keys(d)[0]],
            sentence: d.sentence,
            commit: d.commitment,
            control: d.control_mutuality,
            satis: d.satisfaction,
            trust: d.trust
        }
    });

    dimensions = ["commit", "control", "satis", "trust"]

    const y = {}
    for (i in dimensions) {
        name = dimensions[i]
        y[name] = d3.scaleLinear()
            .domain( [0,1] )
            .range([height, 0])
    }

    x = d3.scalePoint()
        .range([0, width])
        .domain(dimensions);

    function path(d) {
        return d3.line()(dimensions.map(function(p) { return [x(p), y[p](d[p])]; }));
    }

    svg.selectAll("myPath")
        .data(newData)
        .join("path")
            .attr("class", function (d) { return "line line-data-point-" + d.idx + ' data-point-' + d.idx; } )
            .attr("d", path)
            .style("fill", "none" )
            .style("stroke", "lightgrey")
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
            })

    svg.selectAll("myAxis")
        .data(dimensions).enter()
        .append("g")
            .attr("class", "axis")
            .attr("transform", function(d) { return `translate(${x(d)})`; })
        .each(function(d, i) {
            d3.select(this).call(d3.axisLeft().scale(y[d]).ticks(5)
                .tickSize(i === 0 ? 6 : 0)
                .tickPadding(i === 0 ? 3 : 0)
                .tickFormat(i === 0 ? d3.format(".1f") : () => "")
            );
        })
        .append("text")
            .style("text-anchor", "middle")
            .attr("y", -9)
            .text(function(d) { return d; })
            .style("fill", "black")
            .style("cursor", "pointer");

    let clickedAxis = null;

    function drawAxes() {
        svg.selectAll(".axis").remove();

        svg.selectAll("myAxis")
            .data(dimensions).enter()
            .append("g")
                .attr("class", "axis")
                .attr("transform", function(d) { return `translate(${x(d)})`; })
            .each(function(d, i) {
                d3.select(this).call(d3.axisLeft().scale(y[d]).ticks(5)
                    .tickSize(0)
                    .tickPadding(3)
                    .tickFormat(i === 0 ? d3.format(".1f") : "")
                );
            })
            .append("text")
                .style("text-anchor", "middle")
                .attr("y", -9)
                .text(function(d) { return d; })
                .style("fill", "black")
                .style("cursor", "pointer")
                .on("click", function(event, d) {
                    if (!clickedAxis) {
                        clickedAxis = d;
                        d3.select(this).style("fill", "blue");
                    } else if (clickedAxis === d) {
                        d3.select(this).style("fill", "black");
                        clickedAxis = null;
                    } else {
                        const index1 = dimensions.indexOf(clickedAxis),
                            index2 = dimensions.indexOf(d);
                        [dimensions[index1], dimensions[index2]] = [dimensions[index2], dimensions[index1]];

                        x.domain(dimensions);

                        drawAxes();

                        svg.selectAll(".line").attr("d", path);

                        svg.selectAll(".axis text").style("fill", "black");
                        clickedAxis = null;
                    }
                });
    }

    drawAxes();
})

