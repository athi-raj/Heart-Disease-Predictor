```javascript id="zk4wrx"
function toggleMenu() {

    document
    .getElementById("navLinks")
    .classList
    .toggle("active");

}

document
.getElementById("predictionForm")
.addEventListener(
"submit",
async function(e){

    e.preventDefault();

    const formData =
    new FormData(this);

    const data = {

        age:
        formData.get("age"),

        sex:
        formData.get("sex"),

        dataset:
        formData.get("dataset"),

        cp:
        formData.get("cp"),

        trestbps:
        formData.get("trestbps"),

        chol:
        formData.get("chol"),

        fbs:
        formData.get("fbs"),

        restecg:
        formData.get("restecg"),

        thalch:
        formData.get("thalch"),

        exang:
        formData.get("exang"),

        oldpeak:
        formData.get("oldpeak"),

        slope:
        formData.get("slope"),

        ca:
        formData.get("ca"),

        thal:
        formData.get("thal")

    };

    const resultDiv =
    document.getElementById(
        "result"
    );

    resultDiv.innerHTML = `
        <h3>
        Analyzing Patient Data...
        </h3>
    `;

    try{

        const response =
        await fetch(
            "/predict",
            {
                method:"POST",

                headers:{
                    "Content-Type":
                    "application/json"
                },

                body:
                JSON.stringify(data)
            }
        );

        const result =
        await response.json();

        if(result.success){

            let color =
            "#22c55e";

            if(
                result.risk ===
                "Moderate Risk"
            ){
                color =
                "#f59e0b";
            }

            if(
                result.risk ===
                "High Risk"
            ){
                color =
                "#ef4444";
            }

            resultDiv.innerHTML = `

            <h2 style="
            color:${color};
            margin-bottom:15px;
            ">

            ${result.risk}

            </h2>

            <p>

            <strong>
            Confidence:
            </strong>

            ${result.confidence}%

            </p>

            <p>

            <strong>
            Probability:
            </strong>

            ${result.probability}

            </p>

            <br>

            <p>

            AI-generated
            cardiovascular risk
            assessment based on
            the submitted
            clinical parameters.

            </p>

            `;

        }

        else{

            resultDiv.innerHTML = `

            <h3>

            Prediction Failed

            </h3>

            <p>

            ${result.error}

            </p>

            `;

        }

    }

    catch(error){

        resultDiv.innerHTML = `

        <h3>

        Server Error

        </h3>

        <p>

        Unable to connect
        to Flask backend.

        </p>

        `;

        console.error(error);

    }

});
```
