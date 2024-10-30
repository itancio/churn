import plotly.graph_objects as go

def create_gauge_chart(probability, prefix='Churn'):
    # Determine color based on churn probability
    if probability < 0.33:
        color = "green"
    elif probability < 0.66:
        color = "yellow"
    else:
        color = "red"

    # Create gauge chart
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability * 100,  # Adjust to percentage scale
            domain={
                "x": [0, 1],
                "y": [0, 1]
            },
            title={
                "text": prefix + " Probability",
                "font": {
                    "size": 24,
                    "color": "white"
                },
            },
            number={
                "font": {
                    "size": 40,
                    "color": "white"
                }
            },
            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickwidth": 1,
                    "tickcolor": "white"
                },
                "bar": {
                    "color": color
                },
                "bgcolor": "rgba(0, 0, 0, 0)",
                "borderwidth": 2,
                "bordercolor": "white",
                "steps": [
                    {"range": [0, 33], "color": "rgba(0, 255, 0, 0.3)"},
                    {"range": [33, 66], "color": "rgba(255, 255, 0, 0.3)"},
                    {"range": [66, 100], "color": "rgba(255, 0, 0, 0.3)"}
                ],
                "threshold": {
                    "line": {
                        "color": "white",
                        "width": 4
                    },
                    "thickness": 0.75,
                    "value": probability * 100  # Adjust to percentage scale
                }
            }
        )
    )

    # Update chart layout
    fig.update_layout(
        paper_bgcolor="rgba(0, 0, 0, 0)",
        plot_bgcolor="rgba(0, 0, 0, 0)",
        font={"color": "white"},
        width=400,
        height=300,
        margin=dict(l=20, r=20, t=50, b=20)
    )

    return fig

def create_model_probability_chart(probabilities, prefix = 'Churn'):
    models = list(probabilities.keys())
    probs = list(probabilities.values())

    fig = go.Figure(data=[
        go.Bar(
            y=models,
            x=probs,
            orientation="h",
            text=[f"{p: .4%}" for p in probs],
            textposition="auto"
        )
    ])
    fig.update_layout(
        title= prefix + " Probability by Model",
        yaxis_title="Models",
        xaxis_title="Probability",
        xaxis=dict(tickformat=".0%", range=[0, 1]),
        height=400,
        margin=dict(l=20, r=20, t=40, b=20)
    )

    return fig