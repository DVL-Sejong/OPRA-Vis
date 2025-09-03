// 시각화 실행 시간 측정 함수
async function visualize(func, ...args) {
    console.log(`${func.name} started.`);
    const startTime = performance.now(); // 시작 시간

    const result = await func(...args);

    const endTime = performance.now(); // 종료 시간
    const duration = (endTime - startTime) / 1000; // 실행 시간
    console.log(`${func.name} execution time: ${duration.toFixed(2)} seconds.`);

    return result;
}
