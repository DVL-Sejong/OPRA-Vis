function drawCommunicationBehavior(data, similarityMatrix) {
    const slider = initializeSlider(); // 슬라이더 초기화
    initializeEvents(slider); // 이벤트 초기화

    /**
     * 슬라이더를 초기화하고 SVG 요소 추가
     * @returns {d3.Slider} 생성된 d3 슬라이더 객체
     */
    function initializeSlider() {
        const svg = d3.select('#concept-strength').append('svg')
            .attr('width', 840)
            .attr('height', 60);

        const slider = createSlider(); // 슬라이더 생성
        appendSliderGroup(svg, slider); // 슬라이더 그룹 추가
        createCenterText(svg, slider); // 슬라이더 범위 영역 중앙에 텍스트 추가

        return slider;
    }

    /**
     * d3 슬라이더를 생성 및 설정
     * @returns {d3.Slider} 생성된 d3 슬라이더 객체
     */
    function createSlider() {
        // 범위 입력 필드
        const minInput = document.getElementById('concept-strength-min');
        const maxInput = document.getElementById('concept-strength-max');

        return d3.sliderBottom()
            .min(0) // 최소값
            .max(1) // 최대값
            .width(800)
            .ticks(20) // Tick의 개수
            .tickFormat(d3.format('.2f')) // Tick 소수점 아래 두 자리까지 표시
            .default([parseFloat(minInput.value), parseFloat(maxInput.value)]) // 입력 필드의 범위를 기본값으로 사용
            .fill('#2196f3');
    }

    /**
     * SVG 내에 슬라이더 그룹 추가
     * @param {d3.Selection} svg 슬라이더를 추가할 SVG 선택자
     * @param {d3.Slider} slider 추가할 슬라이더 객체
     */
    function appendSliderGroup(svg, slider) {
        const gSlider = svg.append('g').attr('transform', 'translate(20,20)');
        gSlider.call(slider);

        // 핸들 모양을 막대로 변경
        gSlider.selectAll('.handle').attr('d', 'M-2.5,-10 L2.5,-10 L2.5,10 L-2.5,10 Z');
    }

    /**
     * 슬라이더 선택 영역 중앙에 값을 표시하는 텍스트를 생성
     * @param {d3.Selection} svg 텍스트를 추가할 SVG 선택자
     * @param {d3.Slider} slider 슬라이더 객체
     */
    function createCenterText(svg, slider) {
        const gText = svg.append('g').attr('class', 'slider-center-text');
        const rectWidth = 40, rectHeight = 15;
        gText.append('rect') // 배경
            .attr('x', -rectWidth / 2) // 가운데
            .attr('y', -rectHeight / 2) // 가운데
            .attr('width', rectWidth)
            .attr('height', rectHeight)
            .attr('fill', '#333');
        gText.append('text') // 텍스트
            .attr('id', 'num-communication-behavior-data')
            .attr('text-anchor', 'middle')
            .attr('dy', '0.35em')
            .attr('fill', 'white')
            .style('font-size', '.8em')
            .text(0); // 중앙 표시값의 초기값을 0으로 설정
        updateCenterText(slider.default()[0], slider.default()[1]); // 초기 위치 설정
    }

    /**
     * 슬라이더 선택 영역 중앙에 텍스트 위치와 값을 업데이트
     * @param {number} min 슬라이더 선택 영역의 최소값
     * @param {number} max 슬라이더 선택 영역의 최대값
     */
    function updateCenterText(min, max) {
        const centerPosition = ((max - min) / 2 + min) * 800; // 슬라이더 너비 고정값 사용
        const translateX = centerPosition + 20; // 슬라이더의 g 변환값

        d3.select('.slider-center-text')
            .attr('transform', `translate(${translateX}, 20)`);
    }

    /**
     * 이벤트 리스너를 초기화
     * @param {d3.Slider} slider 슬라이더 객체
     */
    function initializeEvents(slider) {
        initializeSectionEvents();
        initializeSliderEvents(slider);
        initializeTableEvents();

        /**
         * 컨테이너 섹션과 관련된 이벤트 리스너를 초기화
         */
        function initializeSectionEvents() {
            const dataTargetConcept = document.getElementById('data-target-concept');
            dataTargetConcept.addEventListener('change', function() {
                // 컨테이너 섹션 헤더 텍스트 변경
                document.getElementById('concept-analysis-header').innerText = this.value;
                // 테이블 업데이트
                updateTable();
                // 스캐터플롯 업데이트
                updateScatterPlot();
            });
        }

        /**
         * 슬라이더와 관련된 이벤트 리스너를 초기화
         * @param {d3.Slider} slider 슬라이더 객체
         */
        function initializeSliderEvents(slider) {
            // 범위 입력 필드
            const minInput = document.getElementById('concept-strength-min');
            const maxInput = document.getElementById('concept-strength-max');

            // 범위 입력 필드 값 변경 시 슬라이더와 스캐터플롯 업데이트
            inputEvent = _ => {
                updateSlider(slider);
                updateScatterPlot();
                
                // 로그 전송
                const concept = document.getElementById('data-target-concept').value;
                sendLog(`Filter updated: ${concept} [${minInput.value}, ${maxInput.value}]`);
            };
            minInput.addEventListener('change', inputEvent);
            maxInput.addEventListener('change', inputEvent);

            // 슬라이더 범위 변경 시 범위 입력 필드 및 중앙 표시값 업데이트
            slider.on('onchange', val => {
                // 범위 입력 필드 업데이트
                minInput.value = val[0];
                maxInput.value = val[1];
                // 슬라이더 중앙 표시값 업데이트
                updateCenterText(val[0], val[1]);
                // 테이블 업데이트
                updateTable();
                // 스캐터플롯 업데이트
                updateScatterPlot();
            });

            slider.on('end', val => {
                // 로그 전송
                const concept = document.getElementById('data-target-concept').value;
                sendLog(`Filter updated: ${concept} [${val[0]}, ${val[1]}]`);
            });
        }

        /**
         * Communication Behavior Data 테이블의 이벤트 리스너를 초기화
         */
        function initializeTableEvents() {
            const tbody = document.getElementById('communication-behavior-data');
            const sectionDecision = document.getElementById('analysis-decision-making');
            const sectionScatterPlot = document.getElementById('scatter-plot');

            tbody.addEventListener('change', function(e) {
                // Analysis 라디오박스 체크 시
                if (e.target && e.target.type === 'radio' && e.target.className === 'analysis-target-radiobutton' && e.target.checked) {
                    document.getElementById('data-target-content-id').value = e.target.getAttribute('data-index');
                    sectionDecision.dispatchEvent(new CustomEvent('selected'));
                    sectionScatterPlot.dispatchEvent(new CustomEvent('selected'));
                }

                // Exclude 체크박스 체크 시
                if (e.target && e.target.type === 'checkbox' && e.target.className === 'exclude-data-checkbox' && e.target.checked) {
                    const targetIndex = e.target.getAttribute('data-index');
                    // const similars = data[targetIndex].similarity
                    const similars = similarityMatrix[targetIndex]
                        .map((similarity, index) => similarity > 0.85 ? index : null)
                        .filter(index => document.getElementById(`exclude-data-${index}`) !== null) // 현재 표에 존재하는 것만
                        .filter(index => index !== null);

                    // 체크한 것과 유사도가 높은 데이터들 제외
                    similars.forEach(similarIndex => {
                        document.getElementById(`exclude-data-${similarIndex}`).checked = true;
                        excludedData[similarIndex] = true; // 설정 업데이트
                    });
                    saveConfigurations(); // 설정 저장

                    // Scatter Plot 갱신 이벤트 호출
                    document.getElementById('scatter-plot').dispatchEvent(new CustomEvent('excluded'));

                    // 로그 전송
                    sendLog(`Excluded record ${targetIndex}: ${data[targetIndex].content}\n- Highly similar records(${similars.toString()}) are also excluded.`);

                    // 메시지 출력
                    alert(`${similars.length} records with high similarity were excluded.`);
                }
            });

            // LLM의 인코딩 결과 갱신 시
            tbody.addEventListener('generated', function(e) {
                const concept = document.getElementById('data-target-concept').value; // 대상 OPRA concept
                const encoding = parseConceptLabelLLM(e.detail.generatedText.split('INPUT: ')[0]); // LLM이 생성한 텍스트를 인코딩

                encodingLLM[e.detail.index][concept] = encoding;

                const encodingText = encoding == null ? 'N/A' : encoding;
                document.getElementById(`encoding-llm-${e.detail.index}`).textContent = encodingText;

                // 설정 저장
                saveConfigurations();

                // 로그 전송
                const groundTruth = data[e.detail.index].opra_label_gt[concept];
                sendLog(`LLM rated ${concept} of contents ID ${e.detail.index} as ${encodingText} (ground-truth = ${groundTruth})`);
            });

            tbody.addEventListener('click', function(e) {
                if (e.target.tagName.toLowerCase() === 'label') {
                    const labelContent = e.target.textContent || e.target.innerText;
                    var wordsToHighlight = labelContent.split(' ').filter(Boolean).map(word => word.toLowerCase());

                    wordCloudEmphasize(wordsToHighlight);
                }
            });
        }
    }

    /**
     * 입력 필드의 값을 기반으로 슬라이더 값을 업데이트
     */
    function updateSlider(slider) {
        // 범위 입력 필드
        const minInput = document.getElementById('concept-strength-min');
        const maxInput = document.getElementById('concept-strength-max');

        // 최소값과 최대값을 범위로 슬라이더 업데이트
        let minVal = parseFloat(minInput.value);
        let maxVal = parseFloat(maxInput.value);

        if (minVal < 0) minVal = 0;
        if (maxVal > 1) maxVal = 1;
        if (minVal > maxVal) minVal = maxVal;

        slider.silentValue([minVal, maxVal]);
        updateCenterText(minVal, maxVal);
    }

    /**
     * Communication Behavior Data 테이블의 내용 업데이트
     */
    function updateTable() {
        const table = document.getElementById('communication-behavior-data');

        // 대상 OPRA concept
        const concept = document.getElementById('data-target-concept').value;

        // 범위 입력 필드
        const weak = document.getElementById('concept-strength-min').value;
        const strong = document.getElementById('concept-strength-max').value;

        // Lasso 필터링
        const lassoActived = document.getElementById('scatter-plot').classList.contains('lasso-actived');

        // 기존에 테이블에 있던 데이터 행을 모두 제거
        while (table.rows.length > 0) {
            table.deleteRow(0);
        }

        // data 배열을 반복하여 테이블 행과 셀을 추가
        let length = 0;
        data.forEach((item, index) => {
            const lassoSelected = document.getElementById(`scatter-${index}`).classList.contains('lasso-selected');
            if ((item.opra[concept] >= weak && item.opra[concept] <= strong) && ((lassoActived && lassoSelected) || !lassoActived)) {
                const row = table.insertRow(); // 새 행(tr) 생성

                const cellAnalysis = row.insertCell(); // Analyze 라디오버튼
                const radiobutton = document.createElement('input');
                radiobutton.type = 'radio';
                radiobutton.id = `analysis-target-${index}`;
                radiobutton.name = 'analysis-target-radiobutton';
                radiobutton.className = 'analysis-target-radiobutton';
                radiobutton.setAttribute('data-index', index);
                cellAnalysis.appendChild(radiobutton);
                
                const cellSentence = row.insertCell(); // Sentence 출력
                const label = document.createElement('label');
                label.setAttribute('for', radiobutton.id);
                label.className = 'target-sentence';
                label.textContent = item.content;
                cellSentence.appendChild(label);

                const cellIntensity = row.insertCell(); // Intensity 출력
                cellIntensity.textContent = item.opra[concept].toFixed(2);

                const cellEncodingLLM = row.insertCell(); // LLM 인코딩 출력
                cellEncodingLLM.id = `encoding-llm-${index}`;
                textEncodingLLM = parseConceptLabel(encodingLLM[index][concept] ? encodingLLM[index][concept] : item.opra_label_llm[concept]);
                cellEncodingLLM.textContent = textEncodingLLM;

                const cellEncodingGT = row.insertCell(); // Ground-truth 인코딩 출력
                cellEncodingGT.textContent = parseConceptLabel(item.opra_label_gt[concept]);

                const cellExclude = row.insertCell(); // 제외 체크박스
                const checkbox = document.createElement('input');
                checkbox.type = 'checkbox';
                checkbox.id = `exclude-data-${index}`;
                checkbox.className = 'exclude-data-checkbox';
                checkbox.checked = excludedData[index];
                checkbox.setAttribute('data-index', index);
                cellExclude.appendChild(checkbox);

                length++;
            }
        });
        
        // 슬라이더에 필터링된 데이터 수 출력
        d3.select('#num-communication-behavior-data').text(length);
    }

    /**
     * Communication Behavior Data 필터에 맞게 스캐터 플롯 업데이트
     */
    function updateScatterPlot() {
        d3.select('#scatter-plot').dispatch('filtered');
    }

    /**
     * LLM의 인코딩 결과 변환
     * @param {*} generatedText 생성된 텍스트
     * @returns 0 혹은 1의 인코딩
     */
    function parseConceptLabelLLM(generatedText) {
        if (generatedText.includes(': True')) {
            return 'True';
        } else if (generatedText.includes(': False')) {
            return 'False';
        } else if (generatedText.includes(': Neutral')) {
            return 'Neutral';
        } else {
            return parseConceptLabel(generatedText);
        }
    }

    function parseConceptLabel(text) {
        const str = (text ?? "").toString().trim();
        if (parseFloat(str) === 0.5) {
            return 'Neutral';
        } else if (parseInt(str) === 1) {
            return 'True';
        } else if (parseInt(str) === 0) {
            return 'False';
        } else {
            return null;
        }
    }
}
