async function drawWordCloud(metadata, scatterPlotOptions, data) {
    const concepts = metadata['concept_column_names']
    const numConcepts = metadata['concepts'].length;
    const svgSize = scatterPlotOptions['svg_size'];
    const margin = scatterPlotOptions['margin'];
    const labelSize = scatterPlotOptions['label_size'];
    const angleSlice = scatterPlotOptions['angle_slice'];
    const labelPositions = scatterPlotOptions['label_positions'];
    const wordCloudSize = {
        width: 170,
        height: 70,
    };
    const svg = d3.select('#svg').append('g')
        .attr('transform', `translate(${parseFloat(svgSize.width / 2 + margin.left)},${parseFloat(svgSize.height / 2 + margin.top)})`);

    // data의 key에 포함된 띄어쓰기를 전부 _로 변경
    const words = Object.fromEntries(
        Object.entries(data).map(([key, value]) => [key.replace(/ /g, '_'), value])
    )

    // 각 레이블의 위치를 기반으로 워드 클라우드 배치
    const tagCloudPromises = [];
    for (let i = 0; i < numConcepts*2; i++) {
        const concept = concepts[i % numConcepts];
        // 레이블 위치 계산
        const angle = angleSlice * i - Math.PI / 2; // 수직 정렬을 위해 조정
        // 회전 각도
        const rotationAngle = angle * 180 / Math.PI + 90;
        // 워드 클라우드 위치 (레이블 위와 아래)
        const labelPosition = labelPositions[i];
        const cloudPositions = [
            {
                x: labelPosition.x,
                y: labelPosition.y - labelSize.height/2 - wordCloudSize.height/2,
            }, {
                x: labelPosition.x,
                y: labelPosition.y - labelSize.height/2 - wordCloudSize.height*1.5,
            }
        ];

        // 긍부정 감정에 대한 워드 클라우드 그리기
        cloudPositions.forEach((pos, idx) => {
            const sentiment = idx === 0 ? 'positive': 'negative';
            const color = sentiment === 'positive' ? '#69b3a2' : 'red';

            const svgGroup = svg.append('g')
                .attr('id', `cloud_${concept}_${sentiment}`)
                .attr('transform', `rotate(${rotationAngle},${labelPosition.x},${labelPosition.y})`);
            
            const conceptLowerCase = concept.toLowerCase().replace(/ /g, '_');
            const label = i < numConcepts ? 'true' : 'false';
            const sentimentShort = idx === 0 ? 'pos' : 'neg';
            const key = `${conceptLowerCase}_${label}_${sentimentShort}`;

            // 비동기적으로 워드 클라우드 그리기
            tagCloudPromises.push(drawTagCloud(svgGroup, words[key], pos.x, pos.y, wordCloudSize.width, wordCloudSize.height, color, sentiment));
        });
    }

    await Promise.all(tagCloudPromises);
}

async function drawTagCloud(svg, wordList, x, y, width, height, color, cls) {
    return new Promise(resolve => {
        const layout = d3.layout.cloud()
            .size([width, height])
            .words(wordList.map(([text, size]) => ({ text, size }))) // 텍스트 크기를 매핑하여 워드 클라우드 데이터 생성
            .padding(1.5)
            .rotate(0) // 회전 설정 (현재 비활성화)
            .fontSize(d => Math.max(7, d.size*2)) // 폰트 크기 설정
            .on('end', words => {
                drawSubCloud(words);
                resolve();
            });
        
        layout.start();
    });

    function drawSubCloud(words) {
        svg.append('g')
            .attr('transform', `translate(${x},${y})`)
            .selectAll('text').data(words).enter()
                .append('text')
                .style('pointer-events', 'none')
                .style('font-size', d => `${d.size}px`) // 폰트 크기 동적 계산
                .style('fill', color)
                .attr('text-anchor', 'middle')
                .attr('class', cls) // 클래스 추가 (positive/negative)
                .attr('transform', d => `translate(${d.x},${d.y})rotate(${d.rotate})`) // 위치 및 회전 설정
                .text(d => d.text);
    }
}

function wordCloudEmphasize(data) {
    d3.selectAll('#svg g text')
        .style('fill', function(d) {
            var cls = this.classList[0];
            if(cls == 'positive') return '#69b3a2';
            else if(cls == 'negative') return 'red';
            else return 'black';
        })

    d3.selectAll("#svg g text").each(function() {
        var textElement = d3.select(this);
        var word = textElement.text().toLowerCase();

        if(data.includes(word)) {
            textElement.style('fill', 'blue');
        }
    })
}