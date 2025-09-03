const barWidth = document.getElementById('opraLabel').clientWidth - margin.left - margin.right;
const barHeight = document.getElementById('opraLabel').clientHeight - margin.left - margin.right;

const opraSvg = d3.select('#opraLabel')
    .append('svg')
        .attr('width', barWidth + margin.left + margin.right)
        .attr('height', barHeight + margin.top + margin.bottom)
    .append('g')
        .attr('transform', `translate(${margin.left},${margin.top})`);

const sentimentSvg = d3.select('#sentLabel')
    .append('svg')
        .attr('width', barWidth + margin.left + margin.right)
        .attr('height', barHeight + margin.top + margin.bottom)
    .append('g')
        .attr('transform', `translate(${margin.left},${margin.top})`);

d3.csv('static/data/unique_column.csv').then(function(data) {
    var commit = 0, control = 0, satis = 0, trust = 0;
    for(var i=0;i<data.length;i++) {
        d = data[i];
        commit += parseInt(d.commitment);
        control += parseInt(d.control_mutuality);
        satis += parseInt(d.satisfaction);
        trust += parseInt(d.trust);
    }

    const percentData = [
        {category: 'Comm.', '0': 100 - commit/data.length * 100, '1': commit/data.length * 100},
        {category: 'Satis.', '0': 100 - satis/data.length * 100, '1': satis/data.length * 100},
        {category: 'Trust.', '0': 100 - trust/data.length * 100, '1': trust/data.length * 100},
        {category: 'C.M.', '0': 100 - control/data.length * 100, '1': control/data.length * 100},
    ];

    const x = d3.scaleBand()
        .range([0, barWidth])
        .padding(0.4)
        .domain(percentData.map(d => d.category));
    const y = d3.scaleLinear()
        .range([barHeight, 0])
        .domain([0, 100]);

    const color = d3.scaleOrdinal()
        .domain(['0', '1'])
        .range(['lightgrey', 'black']);

    const stackKeys = ['0', '1'];

    percentData.forEach(d => {
        let y0 = 0;
        d.segments = stackKeys.map(key => ({key: key, y1: y0, y0: y0 += +d[key]}));
        d.total = d.segments[d.segments.length - 1].y1;
    });

    opraSvg.selectAll(".category")
        .data(percentData)
      .enter().append("g")
        .attr("class", "category")
        .attr("transform", d => `translate(${x(d.category)},0)`)
      .selectAll("rect")
        .data(d => d.segments)
      .enter().append("rect")
        .attr("width", x.bandwidth())
        .attr("y", d => y(d.y0))
        .attr("height", d => y(d.y1) - y(d.y0))
        .style("fill", d => color(d.key));

    opraSvg.append("g")
        .attr("class", "x axis")
        .attr("transform", "translate(0," + barHeight + ")")
        .call(d3.axisBottom(x))
        .style('font-size', '15px');

    const linePoints = percentData.map(d => {
        return {
            x: x(d.category) + x.bandwidth() / 2,
            y: y(d.segments.find(segment => segment.key === '1').y1)
        };
    });

    const lineGenerator = d3.line()
        .x(d => d.x)
        .y(d => d.y)
        .curve(d3.curveMonotoneX);

    opraSvg.append("path")
        .datum(linePoints)
        .attr("fill", "none")
        .attr("stroke", "green")
        .attr("stroke-width", 2)
        .attr("d", lineGenerator);

    opraSvg.append("line")
        .style("stroke", "black")
        .style("stroke-width", 1)
        .attr("x1", 0)
        .attr("y1", y(100))
        .attr("x2", barWidth)
        .attr("y2", y(100));

    opraSvg.append('text')
        .text('OPRA Label Balance')
        .attr('x', barWidth/2)
        .attr('y', 0-10)
        .style('font-size', '20px')
        .style('text-anchor', 'middle');

    opraSvg.append('text')
        .text('target')
        .attr('x', 5)
        .attr('y', -10)
        .style('font-size', '15')
        .style('text-anchor', 'middle')
        .style('fill', 'black')
})

d3.csv('static/data/positive_sentiment.csv').then(function(data) {
    var commit = 0, control = 0, satis = 0, trust = 0;
    var cmSent = 0, cSent = 0, sSent = 0, tSent = 0;
    for(var i=0;i<data.length;i++) {
        d = data[i];
        if(d.commitment == '1') {
            commit += 1;
            cmSent += parseInt(d.sentiment);
        }
        if(d.control_mutuality == '1') {
            control += 1;
            cSent += parseInt(d.sentiment);
        }
        if(d.satisfaction == '1') {
            satis += 1;
            sSent += parseInt(d.sentiment);
        }if(d.trust == '1') {
            trust += 1;
            tSent += parseInt(d.sentiment);
        }
    }

    const percentData = [
        {category: 'Comm.', '0': 100 - cmSent/commit * 100, '1': cmSent/commit * 100},
        {category: 'Satis.', '0': 100 - sSent/satis * 100, '1': sSent/satis * 100},
        {category: 'Trust.', '0': 100 - tSent/trust * 100, '1': tSent/trust * 100},
        {category: 'C.M.', '0': 100 - cSent/control * 100, '1': cSent/control * 100},
    ];

    const x = d3.scaleBand()
        .range([0, barWidth])
        .padding(0.4)
        .domain(percentData.map(d => d.category));
    const y = d3.scaleLinear()
        .range([barHeight, 0])
        .domain([0, 100]);

    const color = d3.scaleOrdinal()
        .domain(['0', '1'])
        .range(['#FF0000', '#4472C4']);

    const stackKeys = ['0', '1'];

    percentData.forEach(d => {
        let y0 = 0;
        d.segments = stackKeys.map(key => ({key: key, y1: y0, y0: y0 += +d[key]}));
        d.total = d.segments[d.segments.length - 1].y1;
    });

    sentimentSvg.selectAll(".category")
        .data(percentData)
      .enter().append("g")
        .attr("class", "category")
        .attr("transform", d => `translate(${x(d.category)},0)`)
      .selectAll("rect")
        .data(d => d.segments)
      .enter().append("rect")
        .attr("width", x.bandwidth())
        .attr("y", d => y(d.y0))
        .attr("height", d => y(d.y1) - y(d.y0))
        .style("fill", d => color(d.key));

    sentimentSvg.append("g")
        .attr("class", "x axis")
        .attr("transform", "translate(0," + barHeight + ")")
        .call(d3.axisBottom(x))
        .style('font-size', '15px');

    sentimentSvg.append("line")
        .style("stroke", "black")
        .style("stroke-width", 1)
        .style("stroke-dasharray", ("3, 3"))
        .attr("x1", 0)
        .attr("y1", y(50))
        .attr("x2", barWidth)
        .attr("y2", y(50));

    sentimentSvg.append("line")
        .style("stroke", "black")
        .style("stroke-width", 1)
        .attr("x1", 0)
        .attr("y1", y(100))
        .attr("x2", barWidth)
        .attr("y2", y(100));

    sentimentSvg.append('text')
        .text('Sentiment Label Balance')
        .attr('x', barWidth/2)
        .attr('y', 0-10)
        .style('font-size', '20px')
        .style('text-anchor', 'middle');

    sentimentSvg.append('text')
        .text('N')
        .attr('x', -10)
        .attr('y', barHeight)
        .style('font-size', '20px')
        .style('text-anchor', 'middle')
        .style('fill', 'red')

    sentimentSvg.append('text')
        .text('P')
        .attr('x', -10)
        .attr('y', 10)
        .style('font-size', '20px')
        .style('text-anchor', 'middle')
        .style('fill', '#4472C4')

    sentimentSvg.append('text')
        .text('target')
        .attr('x', 5)
        .attr('y', barHeight/2-10)
        .style('font-size', '15')
        .style('text-anchor', 'middle')
        .style('fill', 'black')
})