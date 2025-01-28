streamlit_style = """
<style>
    .stAppHeader {
        background-color: rgba(255, 255, 255, 0.0);  /* Transparent background */
        visibility: visible;  /* Ensure the header is visible */
    }
    
    .block-container {
        padding-top: 1rem;
        padding-bottom: 0rem;
        padding-left: 5rem;
        padding-right: 5rem;
    }
    
    .hover-container {
        display: inline-block;
        position: relative;
        cursor: pointer;
    }
    .hover-container:hover {
        background-color: lightgrey;
        border-radius: 5px;
    }
    .tooltip {
        visibility: hidden;
        background-color: white;
        border: 1px solid black;
        color: black;
        text-align: left;
        padding: 5px;
        border-radius: 5px;
        position: absolute;
        z-index: 1;
        top: 30px;
        left: 0;
        box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);
        font-size: 0.9em;
    }
    .tooltip-item {
        padding: 2px 5px;
        border-radius: 3px;
        width: fit-content;
    }
    .tooltip-item:hover {
        background-color: #e0e0e0;
    }
    .hover-container:hover .tooltip {
        visibility: visible;
    }
</style>
"""
