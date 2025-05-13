<style>
.tab {
  display: inline-block;
  margin-right: 20px;
  padding: 10px;
  font-weight: bold;
  cursor: pointer;
  border-bottom: 2px solid transparent;
}

.tab:hover {
  border-bottom: 2px solid #007acc;
}

.tab.active {
  border-bottom: 2px solid #007acc;
}

.tab-content {
  display: none;
  margin-top: 10px;
}

.tab-content.active {
  display: block;
}
</style>

<h1>📘 scRegulate Tutorials</h1>

<p>This page contains multiple tutorials to help you:</p>
<ul>
  <li>Run transcription factor (TF) inference on your own data</li>
  <li>Reproduce the results in our manuscript</li>
</ul>

<div>
  <div class="tab active" onclick="showTab('pbmc')">🧬 PBMC 3K Tutorial</div>
  <div class="tab" onclick="showTab('repro')">📄 Reproduce Paper Results</div>
</div>

<div id="pbmc" class="tab-content active">
  <h2>🧬 TF Inference on PBMC 3K</h2>
  <p>This tutorial walks you through the basic usage of <code>scRegulate</code> on a small PBMC 3K dataset.</p>
  <ul>
    <li><a href="https://github.com/YDaiLab/scRegulate/blob/main/notebooks/tutorial_main.ipynb">View Notebook (.ipynb)</a></li>
    <li><a href="https://ydailab.github.io/scRegulate/tutorial_main.html">View Rendered HTML</a></li>
  </ul>
</div>

<div id="repro" class="tab-content">
  <h2>📄 Reproducing Manuscript Results</h2>
  <p>This notebook replicates the preprocessing and analysis pipeline used in our paper.</p>
  <ul>
    <li><a href="https://github.com/YDaiLab/scRegulate/blob/main/notebooks/Data_Preparation.ipynb">View Notebook (.ipynb)</a></li>
    <li><a href="https://ydailab.github.io/scRegulate/Data_Preparation.html">View Rendered HTML</a></li>
  </ul>
</div>

<script>
function showTab(id) {
  const tabs = document.querySelectorAll('.tab');
  const contents = document.querySelectorAll('.tab-content');
  tabs.forEach(tab => tab.classList.remove('active'));
  contents.forEach(content => content.classList.remove('active'));
  document.querySelector('.tab.active')?.classList.remove('active');
  document.getElementById(id).classList.add('active');
  event.target.classList.add('active');
}
</script>
