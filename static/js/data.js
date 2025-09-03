loadAllData();

async function loadAllData() {
    let metadata = {};
    let scatterPlotOptions = {};
    const scatterPlotData = [];

    try {
        // Metadata of Scatter Plot
        const scatterMetadataResponse = await fetch('/metadata');
        metadata = await scatterMetadataResponse.json();
    } catch (error) {
        console.error('Error fetching metadata:', error)
    }

    try {
        // Scatter Plot
        const scatterPlotResponse = await fetch('/scatter_plot');
        // const scatterPlotData = await scatterPlotResponse.json();
        const reader = scatterPlotResponse.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');

            // 마지막 라인은 불완전할 수 있으므로 buffer에 보관
            buffer = lines.pop() || '';
            
            // JSON 데이터 분할 처리
            for (const line of lines) {
                const trimmed = line.trim();
                if (trimmed) {
                    try {
                        scatterPlotData.push(JSON.parse(trimmed)); 
                    } catch (error) {
                        console.error('Error parsing JSON line:', line, error);
                    }
                }
            }
        }

        // 마지막 버퍼 처리
        if (buffer.trim()) {
            try {
                scatterPlotData.push(JSON.parse(buffer.trim()));
            } catch (error) {
                console.error('Error parsing final JSON line:', buffer, error);
            }
        }

        scatterPlotOptions = await visualize(drawScatterPlot, metadata, scatterPlotData, 'sequential', t => d3.interpolateRdYlBu(1 - t));
    } catch (error) {
        console.error('Error fetching scatter plot data:', error);
    }

    try {
        let startTime = performance.now();

        // Similarity
        const similarityResponse = await fetch('/similarity');

        // 스트리밍된 데이터를 하나의 Uint8Array로 병합
        startTime = performance.now();
        const compressedData = new Uint8Array(await similarityResponse.arrayBuffer());
        console.log(`[Similarity] Chunk merge time: ${((performance.now() - startTime) / 1000).toFixed(2)} seconds.`);

        // 압축 해제
        startTime = performance.now();
        const decompressedData = pako.inflate(compressedData);
        console.log(`[Similarity] Decompression time: ${((performance.now() - startTime) / 1000).toFixed(2)} seconds.`);

        // Uint8Array를 Float32Array로 복원
        startTime = performance.now();
        const similarityMatrix = [];
        const matrixSize = Math.sqrt(decompressedData.length);
        for (let i = 0; i < matrixSize; i++) {
            const row = new Float32Array(matrixSize);
            for (let j = 0; j < matrixSize; j++) {
                const index = i * matrixSize + j;
                row[j] = decompressedData[index] / 255; // 역양자화
            }
            similarityMatrix.push(row);
        }
        console.log(`[Similarity] Matrix conversion time: ${((performance.now() - startTime) / 1000).toFixed(2)} seconds.`);
        
        visualize(drawCommunicationBehavior, scatterPlotData, similarityMatrix);
    } catch (error) {
        console.error('Error fetching similarity matrix:', error);
    }

    visualize(drawDecisionMaking);

    try {
        // Word Cloud
        const sentimentResponse = await fetch('/sentiment');
        const reader = sentimentResponse.body.getReader();
        const decoder = new TextDecoder();
        let data = '';

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            data += decoder.decode(value);
        }
        const sentimentData = await JSON.parse(data);

        visualize(drawWordCloud, metadata, scatterPlotOptions, sentimentData);
    } catch (error) {
        console.error('Error fetching word cloud data:', error);
    }
}

function loadDecisionData(postData, callback) {
    fetch('/decision', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(postData),
    })
    .then(response => response.json())
    .then(callback)
    .catch(error => console.error('Error fetching LLM decision making data:', error))
}

function sendLog(message) {
    fetch('/log', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ message: message }),
    })
}
