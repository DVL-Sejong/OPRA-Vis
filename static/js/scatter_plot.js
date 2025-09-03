function drawScatterPlot(metadata, dataPoints, colorMode='quantize', colorScheme=null) {
    initializeEvents(); // 이벤트 초기화

    const FILTERING_METHOD = 'color'; // 'opacity' or 'color'
    let DISTRIBUTION_CHART_COLOR_MODE = 'stacked'; // 'average', 'gradient', 'stacked'
    let DISTRIBUTION_CHART_SCALE = 'log_10'; // 'linear', 'log_e', 'log_2', 'log_10'

    const scale = metadata['scale'];
    const concepts = metadata['concepts'];
    const numConcepts = concepts.length;

    const scatterPlotSize = { // Size of the scatter plot area
        width: 400,
        height: 400
    };
    const labelSize = { // Size of the label
        width: 170 - (numConcepts - 4) * 40,
        height: 20
    };
    const labelMargin = 50; // Margin between label and SVG
    const labelPadding = 30; // Padding between label and scatter plot
    const labelPositions = [];
    const svgSize = { // Size of the visualization area
        width: scatterPlotSize.width + labelSize.height + labelMargin,
        height: scatterPlotSize.height + labelSize.height + labelMargin
    }

    const margin = {top: 150, left: 150, right: 150, bottom: 150};

    const svg = d3.select("#scatter-plot").append("svg")
        .attr("width", svgSize.width + margin.left + margin.right)
        .attr("height", svgSize.height + margin.top + margin.bottom)
        .attr('id', 'svg');
    const plot = svg.append("g")
        .attr('transform', `translate(${parseFloat(svgSize.width / 2 + margin.left)},${parseFloat(svgSize.height / 2 + margin.top)})`);

    const radius = Math.min(svgSize.width, svgSize.height) / 2 - 40; // Padding for labels
    const angleSlice = Math.PI * 2 / (numConcepts * 2); // Concepts 수의 2배에 해당하는 axes

    plot.selectAll(".data-point")
        .data(dataPoints)
        .enter()
        .append("circle")
        .attr('id', (d, i) => `scatter-${i}`)
        .attr("class", (d, i) => `scatter data-point data-point-${i} scatter-data-point-${i}`)
        .attr("cx", d => d.x * scatterPlotSize.width / 1.8)
        .attr("cy", d => d.y * scatterPlotSize.height / 1.8)
        .attr("r", 3)
        .attr("data-index", (d, i) => i)
        .style("fill", '#555')
        .style('stroke', '#333')
        .style('stroke-width', '1px')
        .style('display', (d, i) => excludedData[i] ? 'none' : 'unset')
        .on("click", function(event, d) {
            resetSelections();
        });

    // 컬러스킴 정의
    let colors = colorScheme;
    if (colorScheme === null) {
        colors = ["black", "blue", "cyan", "lime", "yellow", "orange", "red"]; // 컬러 배열
    }
    
    // 컬러스케일 정의
    const domain = [0, 1]; // 도메인: 0에서 1 사이
    let colorScale;
    if (colorMode === 'quantize') {
        colorScale = d3.scaleQuantize()
            .domain(domain)
            .range(colors);
    } else if (colorMode === 'sequential') {
        colorScale = d3.scaleSequential()
            .domain(domain)
            .interpolator(colors);
    }

    // Draw legend
    drawLegend(plot, colorScale, colorMode);

    // Draw distribution chart for dimension distribution
    drawDistributionChart(plot);

    // Draw labels and axes
    const labelGroup = svg.append('g')
        .attr('class', 'labels')
        .attr('transform', `translate(${parseFloat(svgSize.width / 2 + margin.left)},${parseFloat(svgSize.height / 2 + margin.top)})`);
    for (let i = 0; i < numConcepts*2; i++) {
        const angle = angleSlice * i - Math.PI/2; // Adjust for vertical alignment
        const concept = concepts[i % numConcepts];
        const label = `${concept} = ${(i < numConcepts)? 'True' : 'False'}`;
        const labelPosition = {
            x: (radius + labelPadding) * Math.cos(angle),
            y: (radius + labelPadding) * Math.sin(angle)
        };
        labelPositions.push(labelPosition);

        const rotationAngle = angle * 180 / Math.PI + 90; // Rotate label to make it perpendicular

        const onLabelClicked = function() {
            // 직선 그리기/제거 토글
            toggleLineBetweenOppositeLabels(i, labelPositions, labelGroup);

            // distribution chart 업데이트
            updateDistributionChart(plot, dataPoints, concept, colorScale);

            // 레이블 클릭 시 연관 이벤트 호출
            const dataTargetConcept = document.getElementById("data-target-concept");
            dataTargetConcept.value = concept;
            dataTargetConcept.dispatchEvent(new Event("change"));

            // 데이터 포인트 컬러 업데이트
            updateDataPointColors(concept, dataPoints, plot, colorScale);
            // Communication Behavioral Clues Analysis 활성화/비활성화
            document.getElementById('communication-behavior-default').classList.add('hidden');
            document.getElementById('communication-behavior').classList.remove('hidden');
            // Decision Making 세션 활성화/비활성화
            document.getElementById('analysis-decision-making-default').classList.remove('hidden');
            document.getElementById('analysis-decision-making-loading').classList.add('hidden');
            document.getElementById('analysis-decision-making').classList.add('hidden');

            // 로그 전송
            sendLog(`${scale.toUpperCase()} concept \"${concept}\" enabled`)
        }

        // Label background for better visibility
        labelGroup.append("rect")
            .attr("x", labelPosition.x - labelSize.width / 2)
            .attr("y", labelPosition.y - labelSize.height / 2)
            .attr("width", labelSize.width)
            .attr("height", labelSize.height)
            .attr("class", "label-background")
            .attr("transform", `rotate(${rotationAngle},${labelPosition.x},${labelPosition.y})`)
            .on("click", onLabelClicked);

        // Label text
        labelGroup.append("text")
            .attr("x", labelPosition.x)
            .attr("y", labelPosition.y + 5)
            .attr("class", "label-text")
            .style("text-anchor", "middle")
            .text(label)
            .attr("transform", `rotate(${rotationAngle},${labelPosition.x},${labelPosition.y})`)
            .style("pointer-events", "none")
            .on("click", onLabelClicked);
    }

    var tooltip = d3.select("body").append("div") // TODO: 미사용중
        .attr("class", "tooltip")
        .style("opacity", 0)
        .style("position", "absolute")
        .style("pointer-events", "none")
        .style("padding", "10px")
        .style("background", "rgba(255, 255, 255, 1)")
        .style("border", "1px solid #ccc")
        .style("border-radius", "5px")
        .style("text-align", "left");

    let isLassoActive = false;
    let sliderDivOpen = false;
    let lassoPoints = [];
    const LASSO_HIGHLIGHT = { fill: '#ffcc00', stroke: '#000', r: 5, strokeWidth: '2px' };
    const DEFAULT_POINT   = { r: 3, strokeWidth: '1px' };
    const lassoPath = plot.append("path")
        .attr("stroke", "black")
        .attr("stroke-width", 2)
        .attr("fill", "none")
        .style("pointer-events", "none")
        .style("display", "none");

    d3.select('#svg').on("mousedown", function(event) {
        const tgt = event.target;
        if (tgt && (tgt.classList.contains('label-text') || tgt.classList.contains('label-background'))) return;
        
        isLassoActive = true;
        lassoPoints = [];
        lassoPath.style("display", null);

        document.body.classList.add('no-select');
        event.preventDefault();
        event.stopPropagation();
    });
    const lineGenerator = d3.line();

    d3.select('#svg').on("mousemove", function(event) {
        if (!isLassoActive) return;

        const point = d3.pointer(event, plot.node());

        lassoPoints.push(point);

        lassoPath.attr("d", lineGenerator(lassoPoints));
        event.preventDefault();
    });

    d3.select('#svg').on("mouseup", function(event) {
        const tgt = event.target;
        if (tgt && (tgt.classList.contains('label-text') || tgt.classList.contains('label-background'))) return;
        if (!isLassoActive) return;

        isLassoActive = false;

        if (lassoPoints.length >= 3) {
            selectPointsInsideLasso();
            const selectedCount = plot.selectAll('.lasso-selected').size();
            if (selectedCount > 0 && !sliderDivOpen) {
                displaySlidersWindow(event.pageX, event.pageY);
            }
        }

        lassoPoints = [];
        document.body.classList.remove('no-select');
        event.preventDefault();
    });

    function displaySlidersWindow(mouseX, mouseY) {
        if (sliderDivOpen) {
            return;
        }

        sliderDivOpen = true;

        let slidersDiv = d3.select("body").append("div")
            .attr("class", "sliders-tooltip")
            .style("position", "absolute")
            .style("left", mouseX + "px")
            .style("top", mouseY + "px")
            .style("padding", "10px")
            .style("background", "#f0f0f0")
            .style("border", "1px solid #ccc")
            .style("border-radius", "5px")
            .style("display", "flex")
            .style("flex-direction", "column")
            .style("align-items", "center");

        let maxLabelLength = "CM:".length;

        let sliders = ["t", "s", "cm", "c"];
        sliders.forEach(function(slider) {
        let sliderContainer = slidersDiv.append("div").style("margin", "10px").style("display", "flex").style("align-items", "center");

        sliderContainer.append("label")
            .attr("for", slider + "-slider")
            .style("display", "inline-block")
            .style("width", `${maxLabelLength}em`)
            .text(slider.toUpperCase() + ":")
            .style("text-align", "right")
            .style("margin-right", "10px");

        sliderContainer.append("input")
            .attr("type", "range")
            .attr("min", "0")
            .attr("max", "1")
            .attr("step", "0.01")
            .attr("value", "0.5")
            .attr("id", slider + "-slider")
            .on("input", function() {
                d3.select("#" + slider + "-value").text(this.value);
            });

        sliderContainer.append("span")
            .attr("id", slider + "-value")
            .style("display", "inline-block")
            .style("width", "50px")
            .style("min-width", "50px")
            .style("text-align", "left")
            .text("0.5");
        });

        slidersDiv.append("button")
            .text("Move")
            .style("margin-top", "10px")
            .on("click", function() {
                moveSelectedDots();
                lassoPath.style("display", "none");
                lassoPoints = [];
                isLassoActive = false;
                slidersDiv.remove();
                sliderDivOpen = false;
                resetSelections();
            });

        slidersDiv.append("span")
            .text("✖")
            .style("cursor", "pointer")
            .style("position", "absolute")
            .style("top", "15px")
            .style("right", "10px")
            .on("click", function() {
                resetSelections();
                lassoPath.style("display", "none");
                lassoPoints = [];
                isLassoActive = false;
                slidersDiv.remove();
                sliderDivOpen = false;
                document.body.classList.remove('no-select');
            });

         let dragHeader = slidersDiv.insert("div", ":first-child")
            .attr("class", "drag-header")
            .style("width", "100%")
            .style("padding", "5px")
            .style("background-color", "#ddd")
            .style("cursor", "move")
            .text("Drag here");

        let isDragging = false;
        let dragOffset = { x: 0, y: 0 };

        dragHeader.on("mousedown", function(event) {
            isDragging = true;
            let divTop = parseInt(slidersDiv.style("top"), 10);
            let divLeft = parseInt(slidersDiv.style("left"), 10);
            dragOffset.x = event.clientX - divLeft;
            dragOffset.y = event.clientY - divTop;
            event.preventDefault();
        });

        d3.select(window).on("mousemove", function(event) {
            if (isDragging) {
                slidersDiv.style("left", (event.clientX - dragOffset.x) + "px")
                           .style("top", (event.clientY - dragOffset.y) + "px");
            }
        });

        d3.select(window).on("mouseup", function() {
            isDragging = false;
        });
    }

    function moveSelectedDots() {
        let t = parseFloat(d3.select("#t-slider").property("value")) * 2 - 1;
        let s = parseFloat(d3.select("#s-slider").property("value")) * 2 - 1;
        let c = parseFloat(d3.select("#c-slider").property("value")) * 2 - 1;
        let cm = parseFloat(d3.select("#cm-slider").property("value")) * 2 - 1;

        let moveWeight = 50;

        d3.selectAll('.lasso-selected').each(function() {
                const dot = d3.select(this);
                let cx = parseFloat(dot.attr("cx"));
                let cy = parseFloat(dot.attr("cy"));

                if (cm < 0) cx -= 1 * Math.abs(cm) * moveWeight; else if (cm > 0) cx += 1 * Math.abs(cm) * moveWeight;
                if (c  < 0) { cx -= Math.SQRT1_2 * Math.abs(c) * moveWeight; cy -= Math.SQRT1_2 * Math.abs(c) * moveWeight; }
                else if (c>0){ cx += Math.SQRT1_2 * Math.abs(c) * moveWeight; cy += Math.SQRT1_2 * Math.abs(c) * moveWeight; }
                if (t  > 0) cy -= 1 * Math.abs(t) * moveWeight; else if (t  < 0) cy += 1 * Math.abs(t) * moveWeight;
                if (s  > 0) { cx += Math.SQRT1_2 * Math.abs(s) * moveWeight; cy -= Math.SQRT1_2 * Math.abs(s) * moveWeight; }
                else if (s<0){ cx -= Math.SQRT1_2 * Math.abs(s) * moveWeight; cy += Math.SQRT1_2 * Math.abs(s) * moveWeight; }

                dot.attr("cx", cx).attr("cy", cy);
            });
        }

    function selectPointsInsideLasso() {
        plot.selectAll(".scatter").each(function() {
            const cx = parseFloat(this.getAttribute("cx"));
            const cy = parseFloat(this.getAttribute("cy"));
            const inside = isPointInsidePolygon([cx, cy], lassoPoints);
            d3.select(this).classed("lasso-selected", inside);
        });

        d3.select('#scatter-plot')
            .classed('lasso-actived', true)
            .dispatch('filtered');

        d3.select('#data-target-concept').dispatch('change');
    }

    function isPointInsidePolygon(point, vs) {
        var x = point[0], y = point[1];

        var inside = false;
        for (var i = 0, j = vs.length - 1; i < vs.length; j = i++) {
            var xi = vs[i][0], yi = vs[i][1];
            var xj = vs[j][0], yj = vs[j][1];

            var intersect = ((yi > y) != (yj > y))
                && (x < (xj - xi) * (y - yi) / (yj - yi) + xi);
            if (intersect) inside = !inside;
        }

        return inside;
    }

    function resetSelections() {
        const concept = document.getElementById('data-target-concept')?.value || '';

        plot.selectAll('.data-point')
            .classed('lasso-selected', false)
            .style('stroke-width', '1px');

        if (!concept) {
            plot.selectAll('.data-point')
            .style('fill', '#555')
            .style('stroke', '#333');
        }

        d3.selectAll(".line")
            .style("stroke", "lightgrey")
            .style("opacity", 0.5)
            .style('stroke-width', '1px');

        d3.select('#scatter-plot')
            .classed('lasso-actived', false)
            .dispatch('filtered');

        d3.select('#data-target-concept').dispatch('change');
        plot.selectAll(".highlight-circle").remove();
        lassoPath.style("display", "none");
    }

    // 반대편 라벨 간 직선을 그리거나 제거하는 함수
    function toggleLineBetweenOppositeLabels(labelIndex, labelPositions, labelGroup) {
        const numConcepts = concepts.length;
        
        // 반대편 라벨의 인덱스 계산
        let oppositeIndex;
        if (labelIndex < numConcepts) {
            // 상반부 라벨 -> 하반부 라벨
            oppositeIndex = labelIndex + numConcepts;
        } else {
            // 하반부 라벨 -> 상반부 라벨
            oppositeIndex = labelIndex - numConcepts;
        }
        
        const lineId = `line-${Math.min(labelIndex, oppositeIndex)}-${Math.max(labelIndex, oppositeIndex)}`;
        const existingLine = labelGroup.select(`#${lineId}`);
        
        if (!existingLine.empty()) {
            // 기존 직선이 있으면 제거
            existingLine.remove();
        } else {
            // 먼저 모든 concept 직선 제거
            labelGroup.selectAll('.concept-line').remove();
            
            // 새 직선 그리기
            const startPos = labelPositions[labelIndex];
            const endPos = labelPositions[oppositeIndex];
            
            labelGroup.append("line")
                .attr("id", lineId)
                .attr("class", "concept-line")
                .attr("x1", startPos.x)
                .attr("y1", startPos.y)
                .attr("x2", endPos.x)
                .attr("y2", endPos.y)
                .style("stroke", "#777")
                .style("stroke-width", "1")
                // .style("stroke-dasharray", "5,5")
                .style("opacity", 0.5)
                .style("pointer-events", "none")
                .lower(); // 다른 요소들 뒤로 보내기
        }
    }

    // Communication Behavior Data의 슬라이더 및 Lasso를 통한 필터링 이벤트
    d3.select('#scatter-plot').on("filtered", function() {
        const concept = document.getElementById('data-target-concept')?.value || '';
        const min = parseFloat(document.getElementById('concept-strength-min')?.value ?? '0');
        const max = parseFloat(document.getElementById('concept-strength-max')?.value ?? '1');
        const lassoActived = d3.select(this).classed('lasso-actived');

        plot.selectAll(".data-point").each(function() {
            const sel = d3.select(this);
            const index = sel.attr("data-index");
            const lassoSelected = sel.classed('lasso-selected');

            if (!concept) {
                sel.style('fill', '#555').style('stroke', '#333');
            } 
            else {
                const cocScore = dataPoints[index]?.opra?.[concept];
                if (FILTERING_METHOD === 'opacity') {
                    sel.style("opacity", (cocScore < min || cocScore > max) ? 0.2 : 1);
                    if (cocScore < min || cocScore > max) {
                        sel.style("opacity", 0.2).lower();
                    } 
                    else {
                        sel.style("opacity", 1).raise();
                    }
                } 
                else if (FILTERING_METHOD === 'color') {
                    if (cocScore < min || cocScore > max) {
                        sel.style("fill", "#eee").style("stroke", "#ccc").lower();
                    } 
                    else {
                        const base = colorScale(cocScore);
                        sel.style("fill", base).style("stroke", d3.rgb(base).darker(1)).raise();
                    }
                }
            }

            if (lassoActived && lassoSelected) {
                sel.style("stroke", "#000").raise();
            } 
        });
    });


    // Communication Behavior Data의 데이터 제외 이벤트
    d3.select('#scatter-plot').on('excluded', function() {
        plot.selectAll(".data-point")
            .each(function() {
                const index = d3.select(this).attr("data-index");
                const isExcluded = excludedData[index];
                
                if (isExcluded) { // 제외된 경우
                    // 숨김
                    d3.select(this).classed('hidden', true);
                } else {
                    // 원상복구
                    d3.select(this).classed('hidden', false);
                }
            });
    });

    // 레전드 그리기
    function drawLegend(svg, colorScale, colorMode) {
        const legendWidth = 150; // 레전드의 너비
        const legendHeight = 10; // 레전드의 높이
        const legendX = scatterPlotSize.width - legendWidth - 50, legendY = scatterPlotSize.height - 50; // 레전드의 좌표
    
        // 레전드를 그릴 SVG 그룹 생성
        const legendSvg = svg.append("g")
            .attr("class", "legendQuantize")
            .attr("transform", `translate(${legendX},${legendY})`);

        if (colorMode === 'quantize') {
            // Quantize 컬러 모드: 단계별 색상 사각형
            colorScale.range().forEach((color, i) => {
                const [minValue, maxValue] = colorScale.invertExtent(color);
                const xScale = d3.scaleLinear()
                    .domain(domain)
                    .range([0, legendWidth]);
        
                legendSvg.append("rect")
                    .attr("x", xScale(minValue))
                    .attr("y", 0)
                    .attr("width", xScale(maxValue) - xScale(minValue))
                    .attr("height", legendHeight)
                    .style("fill", color);
            });
        
            // 레전드 아래에 축 추가
            const xScale = d3.scaleLinear()
                .domain(domain)
                .range([0, legendWidth]);
        
            legendSvg.append("g")
                .attr("transform", `translate(0, ${legendHeight})`)
                .call(d3.axisBottom(xScale).ticks(colorScale.range().length).tickFormat(d3.format(".2f")));
        } else if (colorMode === 'sequential') {
            // Sequential 컬러 모드: 연속적인 그라디언트
        
            // 그라디언트 정의
            const defs = svg.select("defs").empty() ? svg.append("defs") : svg.select("defs");
            const gradientId = "color-gradient-" + Math.random().toString(36).substr(2, 9);
            
            const linearGradient = defs.append("linearGradient")
                .attr("id", gradientId)
                .attr("x1", "0%")
                .attr("y1", "0%")
                .attr("x2", "100%")
                .attr("y2", "0%");

            // 그라디언트 스톱 생성
            const numStops = 20;
            for (let i = 0; i <= numStops; i++) {
                const offset = i / numStops;
                const value = domain[0] + offset * (domain[1] - domain[0]);
                linearGradient.append("stop")
                    .attr("offset", `${offset * 100}%`)
                    .attr("stop-color", colorScale(value));
            }

            // 그라디언트로 채워진 사각형 추가
            legendSvg.append("rect")
                .attr("x", 0)
                .attr("y", 0)
                .attr("width", legendWidth)
                .attr("height", legendHeight)
                .style("fill", `url(#${gradientId})`);

            // 레전드 아래에 축 추가
            const xScale = d3.scaleLinear()
                .domain(domain)
                .range([0, legendWidth]);

            legendSvg.append("g")
                .attr("transform", `translate(0, ${legendHeight})`)
                .call(d3.axisBottom(xScale).ticks(5).tickFormat(d3.format(".2f")));
        }
    }

    // 레이블 클릭 시 데이터 포인트의 색상을 업데이트하는 함수
    function updateDataPointColors(concept, dataPoints, svg, colorScale) {
        console.log('Updating the scatter plot...');

        if (this.lastConcept === concept) return;
        this.lastConcept = concept;

        requestAnimationFrame(() => {
            const startTime = performance.now();

            svg.selectAll('.data-point')
                .data(dataPoints)
                .style('fill', d => colorScale(d.opra[concept]))
                // .style('stroke', d => d3.rgb(colorScale(d.opra[concept])).darker(0.5)); // 테두리 색상 - 데이터 포인트 색상보다 살짝 진하게
                .style('stroke', d => d3.rgb(colorScale(d.opra[concept])).darker(1)); // 테두리 색상 - 데이터 포인트 색상보다 살짝 진하게

            console.log(`Scatter plot update time: ${((performance.now() - startTime) / 1000).toFixed(2)} seconds.`);
        });
    }

    function initializeEvents() {
        // Communication Behavior 테이블에서 Analysis 라디오버튼 체크 시 타겟 데이터 선택 이벤트
        document.getElementById('scatter-plot').addEventListener('selected', onTargetSelected);
    }

    /**
     * Communication Behavior Data 선택 시 호출되는 콜백
     */
    function onTargetSelected() {
        const index = document.getElementById('data-target-content-id').value;

        // 선택된 데이터 포인트 가져오기
        const selectedPoint = plot.select(`.data-point-${index}`);
        const cx = parseFloat(selectedPoint.attr('cx'));
        const cy = parseFloat(selectedPoint.attr('cy'));
        const pointRadius = parseFloat(selectedPoint.attr('r'));

        // 기존 강조 효과 제거
        plot.selectAll(".highlight-circle").remove();

        // 데이터 포인트를 감싸는 원 추가
        plot.append("circle")
            .attr("class", "highlight-circle")
            .attr("cx", cx)
            .attr("cy", cy)
            .attr("r", pointRadius + 10) // 데이터포인트보다 10px 크게 설정
            .style("fill", "none")
            .style("stroke", "#FF0000")
            .style("stroke-width", "2px")
            .style("stroke-dasharray", "4 2"); // 점선
    }

    // Distribution chart를 그리는 함수
    function drawDistributionChart(svg) {
        const chartWidth = 120;
        const chartHeight = 40;
        const chartX = -350;
        const chartY = scatterPlotSize.height - 80;
        
        // Distribution chart 그룹 생성
        const distributionChartGroup = svg.append("g")
            .attr("class", "distribution-chart")
            .attr("transform", `translate(${chartX}, ${chartY})`);
        
        // 배경 사각형
        distributionChartGroup.append("rect")
            .attr("class", "distribution-chart-background")
            .attr("x", -10)
            .attr("y", -10)
            .attr("width", chartWidth + 20)
            .attr("height", chartHeight + 30)
            .style("fill", "white")
            // .style("stroke", "#ddd")
            .style("stroke-width", "1px")
            .style("opacity", 0.9);
        
        // 타이틀 (클릭하면 스케일 토글)
        distributionChartGroup.append("text")
            .attr("class", "distribution-chart-title")
            .attr("x", chartWidth / 2)
            .attr("y", -5)
            .style("text-anchor", "middle")
            .style("font-size", "10px")
            .style("fill", "#666")
            .style("cursor", "pointer")
            .text("Axis Distribution")
            .on("click", function() {
                // 스케일 사이클
                const scaleOptions = ['linear', 'log_e', 'log_2', 'log_10'];
                const currentIndex = scaleOptions.indexOf(DISTRIBUTION_CHART_SCALE);
                const nextIndex = (currentIndex + 1) % scaleOptions.length;
                DISTRIBUTION_CHART_SCALE = scaleOptions[nextIndex];
                
                // 현재 concept가 선택되어 있으면 차트 업데이트
                const currentConcept = document.getElementById('data-target-concept').value;
                if (currentConcept) {
                    updateDistributionChart(svg, dataPoints, currentConcept, colorScale);
                }
            });
    }

    // dimension line에 데이터 포인트를 투영하여 위치 계산
    function projectPointsOnDimensionLine(dataPoints, concept) {
        const numConcepts = concepts.length;
        const conceptIndex = concepts.indexOf(concept);
        
        if (conceptIndex === -1) return [];
        
        // concept=1과 concept=0에 해당하는 라벨 위치 찾기
        const concept1Index = conceptIndex; // 상반부 (concept = 1)
        const concept0Index = conceptIndex + numConcepts; // 하반부 (concept = 0)
        
        const concept1Pos = labelPositions[concept1Index];
        const concept0Pos = labelPositions[concept0Index];
        
        // 직선의 방향 벡터 계산
        const lineVector = {
            x: concept1Pos.x - concept0Pos.x,
            y: concept1Pos.y - concept0Pos.y
        };
        
        // 직선 벡터의 길이
        const lineLength = Math.sqrt(lineVector.x * lineVector.x + lineVector.y * lineVector.y);
        
        // 정규화된 방향 벡터
        const normalizedLine = {
            x: lineVector.x / lineLength,
            y: lineVector.y / lineLength
        };
        
        // 각 데이터 포인트를 직선에 투영
        const projectedPositions = dataPoints.map((d, i) => {
            const pointX = d.x * scatterPlotSize.width / 1.8;
            const pointY = d.y * scatterPlotSize.height / 1.8;
            
            // concept=0 위치에서 데이터 포인트까지의 벡터
            const pointVector = {
                x: pointX - concept0Pos.x,
                y: pointY - concept0Pos.y
            };
            
            // 직선 위로의 투영 거리 (내적)
            const projectionDistance = pointVector.x * normalizedLine.x + pointVector.y * normalizedLine.y;
            
            // 0.0-1.0 범위로 정규화 (concept=0에서 0.0, concept=1에서 1.0)
            const normalizedPosition = projectionDistance / lineLength;
            
            return {
                index: i,
                position: Math.max(0.0, Math.min(1.0, normalizedPosition)) // 0.0-1.0 범위로 클램핑
            };
        });
        
        return projectedPositions;
    }

    // Distribution chart를 업데이트하는 함수
    function updateDistributionChart(svg, dataPoints, concept, colorScale) {
        const colorMode = DISTRIBUTION_CHART_COLOR_MODE;
        if (!concept) return;
        
        const chartWidth = 120;
        const chartHeight = 40;
        const distributionChartGroup = svg.select(".distribution-chart");
        
        // 기존 막대 제거
        distributionChartGroup.selectAll(".bar").remove();
        distributionChartGroup.selectAll(".bar-segment").remove();
        distributionChartGroup.selectAll(".bar-border").remove();
        distributionChartGroup.selectAll(".bar-count").remove();
        distributionChartGroup.selectAll(".bar-label").remove();
        distributionChartGroup.selectAll(".axis").remove();
        
        // 기존 bar 그라디언트 제거
        const existingDefs = svg.select("defs");
        if (!existingDefs.empty()) {
            existingDefs.selectAll("[id^='bar-gradient-']").remove();
        }
        
        // dimension line에 투영된 위치들 계산
        const projectedPositions = projectPointsOnDimensionLine(dataPoints, concept);
        
        if (projectedPositions.length === 0) return;
        
        // 데이터 분포 계산
        const numBins = 20;
        const minVal = 0.0;
        const maxVal = 1.0;
        const binWidth = (maxVal - minVal) / numBins;
        
        // 히스토그램 데이터 생성
        const bins = Array(numBins).fill(0).map((_, i) => ({
            binStart: minVal + i * binWidth,
            binEnd: minVal + (i + 1) * binWidth,
            count: 0,
            midpoint: minVal + (i + 0.5) * binWidth,
            conceptValues: [] // 실제 concept 값들 저장
        }));
        
        // 투영된 위치를 구간별로 분류하고 실제 concept 값도 저장
        projectedPositions.forEach(({ position, index }) => {
            let binIndex = Math.floor((position - minVal) / binWidth);
            if (binIndex >= numBins) binIndex = numBins - 1; // 경계값 처리 (1.0인 경우)
            bins[binIndex].count++;
            
            // 해당 데이터 포인트의 실제 concept 값 저장
            const conceptValue = dataPoints[index].opra[concept];
            if (conceptValue !== undefined) {
                bins[binIndex].conceptValues.push(conceptValue);
            }
        });
        
        // 각 bin의 평균, 최소, 최대 concept 값 계산
        bins.forEach(bin => {
            if (bin.conceptValues.length > 0) {
                bin.averageConceptValue = bin.conceptValues.reduce((sum, val) => sum + val, 0) / bin.conceptValues.length;
                bin.minConceptValue = Math.min(...bin.conceptValues);
                bin.maxConceptValue = Math.max(...bin.conceptValues);
            } else {
                bin.averageConceptValue = bin.midpoint; // 데이터가 없으면 midpoint 사용
                bin.minConceptValue = bin.midpoint;
                bin.maxConceptValue = bin.midpoint;
            }
        });
        
        const maxCount = Math.max(...bins.map(b => b.count));
        if (maxCount === 0) return;
        
        // 스케일 설정
        const xScale = d3.scaleLinear()
            .domain([minVal, maxVal])
            .range([0, chartWidth]);
        
        // Y축 스케일 설정
        let yScale;
        if (DISTRIBUTION_CHART_SCALE.startsWith('log_')) {
            // 로그 스케일의 경우 0값을 피하기 위해 최소값을 1로 설정
            const logDomain = [Math.max(1, Math.min(...bins.filter(b => b.count > 0).map(b => b.count))), maxCount];
            
            let logBase;
            switch (DISTRIBUTION_CHART_SCALE) {
                case 'log_e':
                    logBase = Math.E;
                    break;
                case 'log_2':
                    logBase = 2;
                    break;
                case 'log_10':
                    logBase = 10;
                    break;
                default:
                    logBase = 10; // 기본값
            }
            
            yScale = d3.scaleLog()
                .base(logBase)
                .domain(logDomain)
                .range([chartHeight, 0]);
        } else {
            yScale = d3.scaleLinear()
                .domain([0, maxCount])
                .range([chartHeight, 0]);
        }
        
        // 막대 그리기
        const bars = distributionChartGroup.selectAll(".bar")
            .data(bins)
            .enter()
            .append("rect")
            .attr("class", "bar")
            .attr("x", d => xScale(d.binStart))
            .attr("y", d => {
                if (DISTRIBUTION_CHART_SCALE.startsWith('log_') && d.count === 0) {
                    return chartHeight; // 0인 경우 바닥에 배치
                }
                return yScale(Math.max(1, d.count));
            })
            .attr("width", Math.max(1, chartWidth / numBins - 1))
            .attr("height", d => {
                if (DISTRIBUTION_CHART_SCALE.startsWith('log_') && d.count === 0) {
                    return 0; // 0인 경우 높이 0
                }
                return chartHeight - yScale(Math.max(1, d.count));
            })
            .style("stroke", "#bbb")
            .style("stroke-width", "0.5px");

        // 색상 모드에 따른 fill 적용
        if (colorMode === 'average') {
            // 평균값 색상 모드
            bars.style("fill", d => colorScale(d.averageConceptValue));
        } else if (colorMode === 'stacked') {
            // 스택 색상 모드
            bars.style("fill", "none"); // 기본 fill 제거
            
            // 각 막대에 대해 스택된 색상 세그먼트 생성
            bins.forEach((binData, binIndex) => {
                if (binData.count === 0) return;
                
                // concept 값을 더 세밀한 구간으로 분류
                const numColorBins = 5; // 색상 구간 수
                const minVal = Math.min(...binData.conceptValues);
                const maxVal = Math.max(...binData.conceptValues);
                const colorBinWidth = maxVal > minVal ? (maxVal - minVal) / numColorBins : 0;
                
                const colorGroups = {};
                binData.conceptValues.forEach(value => {
                    let colorBinIndex;
                    if (colorBinWidth === 0) {
                        colorBinIndex = 0;
                    } else {
                        colorBinIndex = Math.min(Math.floor((value - minVal) / colorBinWidth), numColorBins - 1);
                    }
                    const representativeValue = minVal + (colorBinIndex + 0.5) * colorBinWidth;
                    const color = colorScale(representativeValue);
                    
                    if (!colorGroups[color]) {
                        colorGroups[color] = { count: 0, value: representativeValue };
                    }
                    colorGroups[color].count++;
                });
                
                const effectiveCount = Math.max(1, binData.count);
                const totalHeight = chartHeight - yScale(effectiveCount);
                const barX = xScale(binData.binStart);
                const barWidth = Math.max(1, chartWidth / numBins - 1);
                
                let currentY = yScale(effectiveCount);
                
                // 색상 그룹을 값에 따라 내림차순 정렬 (높은 값이 위에)
                const sortedColorGroups = Object.entries(colorGroups)
                    .sort(([, a], [, b]) => b.value - a.value);
                
                // 각 색상 그룹별로 세그먼트 생성
                sortedColorGroups.forEach(([color, groupData]) => {
                    const segmentHeight = (groupData.count / binData.count) * totalHeight;
                    
                    distributionChartGroup.append("rect")
                        .attr("class", "bar-segment")
                        .attr("x", barX)
                        .attr("y", currentY)
                        .attr("width", barWidth)
                        .attr("height", segmentHeight)
                        .style("fill", color)
                        .style("stroke", "none");
                    
                    currentY += segmentHeight;
                });
                
                // 전체 막대에 테두리 추가
                distributionChartGroup.append("rect")
                    .attr("class", "bar-border")
                    .attr("x", barX)
                    .attr("y", yScale(effectiveCount))
                    .attr("width", barWidth)
                    .attr("height", totalHeight)
                    .style("fill", "none")
                    .style("stroke", "#bbb")
                    .style("stroke-width", "0.5px");
            });
        } else {
            // 그라디언트 색상 모드
            const defs = svg.select("defs").empty() ? svg.append("defs") : svg.select("defs");
            
            bars.style("fill", (d, i) => {
                if (d.count === 0 || d.minConceptValue === d.maxConceptValue) {
                    return colorScale(d.averageConceptValue);
                }
                
                // 각 막대별 그라디언트 ID
                const gradientId = `bar-gradient-${i}`;
                
                // 기존 그라디언트 제거
                defs.select(`#${gradientId}`).remove();
                
                // 새 그라디언트 생성
                const gradient = defs.append("linearGradient")
                    .attr("id", gradientId)
                    .attr("x1", "0%")
                    .attr("y1", "100%") // 아래쪽에서 시작
                    .attr("x2", "0%")
                    .attr("y2", "0%");  // 위쪽으로
                
                // 아래쪽 색상 (최소값)
                gradient.append("stop")
                    .attr("offset", "0%")
                    .attr("stop-color", colorScale(d.minConceptValue));
                
                // 위쪽 색상 (최대값)
                gradient.append("stop")
                    .attr("offset", "100%")
                    .attr("stop-color", colorScale(d.maxConceptValue));
                
                return `url(#${gradientId})`;
            });
        }

        // 막대 위에 개수 표시 (마우스오버 시에만)
        distributionChartGroup.selectAll(".bar-count")
            .data(bins.filter(d => d.count > 0)) // 0보다 큰 값만 표시
            .enter()
            .append("text")
            .attr("class", "bar-count")
            .attr("x", d => xScale(d.binStart) + (chartWidth / numBins - 1) / 2)
            .attr("y", d => yScale(d.count) - 2)
            .style("text-anchor", "middle")
            .style("font-size", "8px")
            .style("fill", "#333")
            .style("opacity", 0) // 기본적으로 숨김
            .text(d => d.count);

        // 막대에 마우스 이벤트 추가
        distributionChartGroup.selectAll(".bar")
            .on("mouseover", function(event, d) {
                // 해당 막대의 count 텍스트만 표시
                distributionChartGroup.selectAll(".bar-count")
                    .filter(textData => textData === d)
                    .style("opacity", 1);
            })
            .on("mouseout", function(event, d) {
                // 모든 count 텍스트 숨김
                distributionChartGroup.selectAll(".bar-count")
                    .style("opacity", 0);
            });
        
        // X축 추가
        const xAxis = d3.axisBottom(xScale)
            .tickValues([0, 1])
            .tickFormat(d => d === 0 ? "False" : "True");
        
        distributionChartGroup.append("g")
            .attr("class", "axis x-axis")
            .attr("transform", `translate(0, ${chartHeight})`)
            .call(xAxis);
        
        // Y축 추가  
        let yAxis;
        if (DISTRIBUTION_CHART_SCALE.startsWith('log_')) {
            // 로그 스케일에서는 더 적은 틱 사용하고 최소/최대값만 표시
            const domain = yScale.domain();
            const tickValues = [domain[0], domain[1]]; // 최소값과 최대값만 표시
            
            yAxis = d3.axisLeft(yScale)
                .tickValues(tickValues)
                .tickFormat(d3.format("d"));
        } else {
            yAxis = d3.axisLeft(yScale)
                .ticks(3)
                .tickFormat(d3.format("d"));
        }
        
        distributionChartGroup.append("g")
            .attr("class", "axis y-axis")
            .call(yAxis);
        
        // 컨셉 이름으로 타이틀 업데이트
        distributionChartGroup.select(".distribution-chart-title")
            .text(`${concept} Distribution`);
            // .text("Axis Distribution");
    }

    return {
        'svg_size': svgSize,
        'margin': margin,
        'label_size': labelSize,
        'angle_slice': angleSlice,
        'label_positions': labelPositions,
    };
}
