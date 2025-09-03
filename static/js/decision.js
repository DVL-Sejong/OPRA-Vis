function drawDecisionMaking() {
    initializeEvents(); // 이벤트 초기화

    /**
     * 이벤트 리스너를 초기화
     */
    function initializeEvents() {
        // Analysis 라디오버튼 체크 시 타겟 데이터 선택 이벤트
        document.getElementById('analysis-decision-making').addEventListener('selected', onTargetSelected);

        // 프롬프트 수정 버튼 클릭 이벤트
        document.getElementById('edit-prompt-button').addEventListener('click', _ => {
            // 현재 프롬프트 출력
            const index = document.getElementById('data-target-content-id').value;
            const concept = document.getElementById('data-target-concept').value;
            document.getElementById('prompt-edit-textarea').value = prompts[index][concept].prompt;

            // 프롬프트 수정 모달 활성화
            document.getElementById('prompt-edit-modal').classList.remove('hidden');
        });
        document.getElementById('save-prompt-button').addEventListener('click', _ => {
            // 프롬프트 저장
            const index = document.getElementById('data-target-content-id').value;
            const concept = document.getElementById('data-target-concept').value;
            const modifiedPrompt = document.getElementById('prompt-edit-textarea').value;
            const originalPrompt = document.getElementById('original-prompt').value;

            if (!modifiedPrompt.includes("{content}")) {
                alert("Please ensure to include the {content} placeholder in your prompt, as it is crucial for inserting the data to be assessed.");
                return;
            }

            if (modifiedPrompt != originalPrompt) {
                prompts[index][concept].prompt = modifiedPrompt;
                prompts[index][concept].excuted = false;
                document.getElementById('run-assessment-notice').classList.remove('hidden');
                document.getElementById('run-assessment-button').disabled = false;

                sendLog(`Modified the prompt of the concept ${concept}`)
            } else {
                prompts[index][concept].excuted = true;
                document.getElementById('run-assessment-notice').classList.add('hidden');
                document.getElementById('run-assessment-button').disabled = true;
            }

            // 프롬프트 수정 모달 비활성화
            document.getElementById('prompt-edit-modal').classList.add('hidden');
        });
        document.getElementById('cancel-prompt-button').addEventListener('click', _ => {
            // 프롬프트 수정 모달 비활성화
            document.getElementById('prompt-edit-modal').classList.add('hidden');
        });

        // 평가 실행 버튼 클릭 이벤트
        document.getElementById('run-assessment-button').addEventListener('click', _ => {
            const index = document.getElementById('data-target-content-id').value;
            const concept = document.getElementById('data-target-concept').value;

            if (!prompts[index][concept].prompt.includes("{content}")) {
                alert("Please ensure to include the {content} placeholder in your prompt, as it is crucial for inserting the data to be assessed.");
                return;
            }

            // 세션 비활성화
            document.getElementById('analysis-decision-making-default').classList.add('hidden');
            document.getElementById('analysis-decision-making-loading').classList.remove('hidden');
            document.getElementById('analysis-decision-making').classList.add('hidden');

            // 수정된 프롬프트로 LLM의 decision 데이터 로드
            loadDecisionData({
                'concept': concept,
                'content_id_list': [parseInt(index)],
                'prompt': prompts[index][concept].prompt,
            }, onDecisionDataLoaded);
        })
    }

    /**
     * Communication Behavior Data 선택 시 호출되는 콜백
     */
    function onTargetSelected() {
        const index = document.getElementById('data-target-content-id').value;
        const concept = document.getElementById('data-target-concept').value || "Trust"; // 기본값 "Trust"
        const targets = document.getElementsByClassName('analysis-target-radiobutton');
        const contentIdList = Array.from(targets)
            .filter(target => target.checked)
            .map(target => parseInt(target.getAttribute('data-index')));
        
        // 로딩 메시지 활성화
        document.getElementById('analysis-decision-making-default').classList.add('hidden');
        document.getElementById('analysis-decision-making-loading').classList.remove('hidden');
        document.getElementById('analysis-decision-making').classList.add('hidden');

        // 다른 데이터 선택 비활성화
        [...document.getElementsByClassName('analysis-target-radiobutton')].forEach(radio => radio.disabled = true);

        // LLM의 decision 데이터 로드
        loadDecisionData({
            'concept': concept,
            'content_id_list': contentIdList,
            'prompt': prompts[index][concept] == null ? null : prompts[index][concept].prompt,
        }, onDecisionDataLoaded);
    }

    /**
     * LLM의 decision 데이터가 로드되었을 때 호출되는 콜백
     * @param {*} data LLM의 decision 데이터
     */
    function onDecisionDataLoaded(data) {
        const index = data.data[0]['content_id'];
        const concept = document.getElementById('data-target-concept').value;
        prompts[index][concept].prompt = data.data[0]['prompt_text'];
        prompts[index][concept].excuted = true;
        document.getElementById('original-prompt').value = data.data[0]['prompt_text'];

        // 설정 저장
        saveConfigurations();

        applyDataToPromptTable(data);
        applyDataToISATable(data);
        
        // 섹션 활성화
        document.getElementById('analysis-decision-making-default').classList.add('hidden');
        document.getElementById('analysis-decision-making-loading').classList.add('hidden');
        document.getElementById('analysis-decision-making').classList.remove('hidden');

        // 버튼 활성화
        document.getElementById('edit-prompt-button').disabled = false;
        document.getElementById('run-assessment-button').disabled = true;
        document.getElementById('run-assessment-notice').classList.add('hidden');

        // 다른 데이터 선택 활성화
        [...document.getElementsByClassName('analysis-target-radiobutton')].forEach(radio => radio.disabled = false);

        // Communication Behavior Data 이벤트 호출
        document.getElementById('communication-behavior-data').dispatchEvent(new CustomEvent('generated', {
            detail: {
                index: index,
                generatedText: data.data[0]['generated_text'],
            }
        }));
    }
    
    /**
     * 프롬프트 테이블에 데이터 적용
     * @param {*} data LLM 반환 데이터
     */
    function applyDataToPromptTable(data) {
        const tbody = document.getElementById('decision-prompt-data');
        const numPromptSentences = data.data[0]['num_prompt_sentences'];

        // 테이블 내용 초기화
        while (tbody.firstChild) {
            tbody.removeChild(tbody.firstChild);
        }

        // 테이블 내용 생성
        data.data[0]['sentences'].filter((_, index) => index < numPromptSentences).forEach((item, index) => {
            const tr = tbody.insertRow(); // 행 추가
            
            const tdId = tr.insertCell();
            tdId.className = 'id-cell';
            tdId.textContent = index;

            const tdSentence = tr.insertCell();
            tdSentence.className = 'sentence-cell';
            tdSentence.textContent = item['sentence'];
        });
    }

    /**
     * ISA 시각화 테이블에 데이터 적용
     * @param {*} data LLM 반환 데이터
     */
    function applyDataToISATable(data) {
        const tbody = document.getElementById('decision-isa-data');
        const numPromptSentences = data.data[0]['num_prompt_sentences'];
        
        // 테이블 내용 초기화
        while (tbody.firstChild) {
            tbody.removeChild(tbody.firstChild);
        }

        // 컬러 스케일 정의
        const customColors = ["black", "blue", "cyan", "lime", "yellow", "orange", "red"];
        // const colorScale = d3.scaleLinear()
        //     .domain(customColors.map((_, i) => i / (customColors.length - 1))) // 도메인: 0.0에서 1.0 사이
        //     .range(customColors); // 범위: 컬러 배열
        const domain = [0, 1];
        const colorScale = d3.scaleQuantize()
            .domain(domain)
            .range(customColors);

        // ISA 셀 헤더 가로 병합
        document.getElementById('decision-isa-header').setAttribute('colspan', data.data[0]['sentences'].length);

        // 테이블 내용 생성
        data.data[0]['sentences'].filter((_, index) => index >= numPromptSentences).forEach((item, index) => {
            const tr = tbody.insertRow(); // 행 추가
            
            const tdId = tr.insertCell();
            tdId.className = 'id-cell';
            tdId.textContent = index + numPromptSentences;

            const tdSentence = tr.insertCell(); // 문장 출력 셀 추가
            tdSentence.className = 'sentence-cell';
            tdSentence.setAttribute('colspan', 2);
            tdSentence.textContent = item['sentence'];
            
            // ISA 시각화 셀 추가
            item['isa'].filter((_, isaIndex) => isaIndex <= index + numPromptSentences).forEach((isaScore, isaIndex) => {
                const tdISA = tr.insertCell(); // 셀 생성
                tdISA.classList.add('isa-cell', 'isa-wrapper');

                const circle = document.createElement('span'); // ISA 시각화의 데이터포인트 생성
                circle.className = 'isa-point';
                circle.style.backgroundColor = colorScale(isaScore);
                circle.addEventListener('hover', _ => {
                    // 로그 전송
                    sendLog(`Drilldown looked up\n- [generated=${index+numPromptSentences}] ${item['sentence']}\n- [focused=${isaIndex}] ${data.data[0]['sentences'][isaIndex]['sentence']}\n- Attention: ${isaScore}`)
                });
                tdISA.append(circle);

                const drilldown = createISADrillDown( // 드릴다운 생성
                    generatedIndex = index + numPromptSentences,
                    generatedSentence = item['sentence'],
                    focusedIndex = isaIndex,
                    focusedSentence = data.data[0]['sentences'][isaIndex]['sentence'],
                    isaScore = isaScore,
                );
                tdISA.append(drilldown);
            })

            // 나머지 개수만큼 빈 ISA 셀 추가
            for (let i = 0; i < data.data[0]['sentences'].length - index - numPromptSentences - 1; i++) {
                const tdISA = tr.insertCell();
                tdISA.className = 'isa-cell';
            }
        });

        // 레전드 영역 초기화
        const trLegend = document.getElementById('decision-isa-legend');
        const tdLegend = document.getElementById('decision-isa-legend-cell');
        while (trLegend.children.length > 2) { // 레전드 행 초기화
            trLegend.removeChild(trLegend.lastChild);
        }
        tdLegend.textContent = 'Attention: '; // 레전드 셀 초기화

        // 레전드 추가
        const legendWidth = 225;
        const legendHeight = 35;
        const legendContainer = d3.select(tdLegend).append('svg')
            .attr('width', legendWidth)
            .attr('height', legendHeight)
            .style('vertical-align', 'middle');
        
        // 경계 지점 계산
        const ticks = colorScale.range().map((color) => {
            const invert = colorScale.invertExtent(color);
            return invert[0];
        });
        // 마지막 경계값 추가
        ticks.push(domain[1]);
        
        // 레전드 사각형 그리기
        colorScale.range().forEach((color, i) => {
            legendContainer.append('rect')
                .attr('x', (ticks[i] * legendWidth))
                .attr('y', 0)
                .attr('width', (ticks[i+1] - ticks[i]) * legendWidth)
                .attr('height', 15)
                .style('fill', color);
        });
        
        // 경계 지점에 눈금 표시
        ticks.forEach((tick, i) => {
            legendContainer.append('line')
                .attr('x1', tick * legendWidth)
                .attr('x2', tick * legendWidth)
                .attr('y1', 15)
                .attr('y2', 20)
                .style('stroke', 'black');
        
            const textAnchor = i === 0 ? 'start' : i === ticks.length - 1 ? 'end' : 'middle';
            const tickText = parseFloat(tick.toFixed(2)).toString(); // 소수점 이하가 0이면 제거됨
            legendContainer.append('text')
                .attr('x', tick * legendWidth)
                .attr('y', 35)
                .attr('text-anchor', textAnchor)
                .style('font-size', '.8em')
                .text(tickText);
        });

        // Axis 추가
        for (let i = 0; i < data.data[0]['sentences'].length; i++) {
            tdAxisItem = trLegend.insertCell();
            tdAxisItem.className = 'isa-cell';
            tdAxisItem.textContent = i;

            // 프롬프트 문장과 생성된 문장을 구분하는 컬러 적용
            tdAxisItem.classList.add(i < numPromptSentences ? 'prompt' : 'generated');
        }
    }

    /**
     * ISA 시각화의 드릴다운 생성
     * @param {number} generatedIndex 생성 문장의 인덱스
     * @param {string} generatedSentence 생성 문장의 내용
     * @param {number} focusedIndex 포커스된 문장의 인덱스
     * @param {string} focusedSentence 포커스된 문장의 내용
     * @param {number} isaScore 문장 간 어텐션 스코어
     * @returns 드릴다운 DOM element
     */
    function createISADrillDown(generatedIndex, generatedSentence, focusedIndex, focusedSentence, isaScore) {
        const template = document.getElementById('decision-isa-drilldown-template'); // 드릴다운의 템플릿
        const drilldown = document.createElement('div'); // 생성할 드릴다운

        drilldown.className = 'isa-drilldown';
        drilldown.innerHTML = template.innerHTML; // 템플릿의 양식 복사
        drilldown.getElementsByClassName('generated-sentence-id')[0].textContent = generatedIndex;
        drilldown.getElementsByClassName('generated-sentence-content')[0].textContent = generatedSentence;
        drilldown.getElementsByClassName('focused-sentence-id')[0].textContent = focusedIndex;
        drilldown.getElementsByClassName('focused-sentence-content')[0].textContent = focusedSentence;
        drilldown.getElementsByClassName('inter-sentence-attention-score')[0].textContent = isaScore.toFixed(5);

        return drilldown;
    }
}
