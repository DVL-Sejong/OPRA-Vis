// LLM의 인코딩 결과
let encodingLLM = {};
let prompts = {};
let excludedData = {};

initializeConfigurations();
loadConfigurations();

function initializeConfigurations() {
    for (let i = 0; i < 10000; i++) {
        encodingLLM[i] = {
            'Trust': null,
            'Commitment': null,
            'Control Mutuality': null,
            'Satisfaction': null,
            'Threat': null,
        };
        prompts[i] = {
            'Trust': {
                prompt: null,
                excuted: false,
            },
            'Commitment': {
                prompt: null,
                excuted: false,
            },
            'Control Mutuality': {
                prompt: null,
                excuted: false,
            },
            'Satisfaction': {
                prompt: null,
                excuted: false,
            },
            'Threat': {
                prompt: null,
                excuted: false,
            },
        };
        excludedData[i] = false;
    }
}

function loadConfigurations() {
    fetch('/configs')
        .then(response => {
            if (!response.ok)
                throw new Error('No configurations found');
            return response.json();
        })
        .then(data => {
            if (data.encodingLLM)
                encodingLLM = data.encodingLLM;
            if (data.prompts)
                prompts = data.prompts;
            if (data.excludedData)
                excludedData = data.excludedData;
        })
        .catch(error => console.error('Error fetching configurations:', error));
}

function saveConfigurations() {
    const data = {
        encodingLLM: encodingLLM,
        prompts: prompts,
        excludedData: excludedData,
    };

    fetch('/configs', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(data),
    })
    .then(response => response.json())
    .then(data => {
        console.log(data.message);
    })
    .catch(error => console.error('Error saving configurations:', error));
}
