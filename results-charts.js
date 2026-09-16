(function () {
  'use strict';
  const data = window.Bridge3DExperiments;
  const escape = value => String(value).replace(/[&<>"']/g, char => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[char]));
  const color = key => data.palette[key] || key;
  const number = (value, decimals = 1) => Number(value).toFixed(decimals).replace(/\.0$/, '');

  function specification(kind, choice) {
    const common = {kind, yMax:100, yTicks:[0,25,50,75,100], yTitle:'Success rate (%)', unit:'%', directLabels:true};
    if (kind === 'real') return {...common, ...data.real, title:'Real world experiments', directLabels:false};
    if (kind === 'efficiency') return {...common, ...data.efficiency, title:'Data efficiency', xTitle:'Training demonstrations'};
    if (kind === 'spatial') return {...common, ...data.spatial, title:'Spatial generalization', xTitle:'Height offset (cm)'};
    if (kind === 'semantics') return {...common, ...data.semantics, title:'Point semantics & encoder'};
    if (kind === 'components') {
      const index = choice === undefined ? 5 : Number(choice);
      return {...common, title:'Key component ablation', xTitle:data.components.categories[index], categories:['w/o IF & EC','w/o EC','w/o IF','Full Bridge3D'], shortLabels:[['w/o','IF & EC'],['w/o EC'],['w/o IF'],['Full','Bridge3D']], series:[{name:'Success rate',values:data.components.series.map(s=>s.values[index]),barColors:data.components.series.map(s=>s.color)}], legend:data.components.series.map(s=>({name:s.name,color:s.color}))};
    }
    if (kind === 'probe') {
      const index = choice === undefined ? 4 : Number(choice);
      const selected = data.probe.series[index];
      return {kind,title:'Layer-wise linear probe',categories:selected.rows.map(row=>String(row.block)),series:[{name:`t = ${selected.step}`,color:data.probePalette[index],values:selected.rows.map(row=>row.mean),lower:selected.rows.map(row=>row.lower),upper:selected.rows.map(row=>row.upper)}],yMax:8,yTicks:[0,2,4,6,8],yTitle:'Error (×10⁻²)',unit:' ×10⁻²',xTitle:'Block index',directLabels:false,ticks:[0,5,10,15,20,25,28],highlight:[7,11],choice:index};
    }
    throw new Error(`Unknown experiment: ${kind}`);
  }

  function render(kind, choice) {
    const spec = specification(kind, choice);
    const W=600, H=382, left=51, right=13, top=40, bottom=300;
    const width=W-left-right, height=bottom-top;
    const groupWidth=width/spec.categories.length;
    const groupInner=groupWidth*(spec.categories.length>10 ? .70 : .76);
    const count=spec.series.length;
    const gap=count>1 ? Math.min(4,groupInner*.055) : 0;
    const barWidth=Math.min(58,(groupInner-gap*(count-1))/count);
    const groupSpan=barWidth*count+gap*(count-1);
    const y=value=>bottom-value/spec.yMax*height;
    const out=[`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${W} ${H}" class="result-svg" role="group" aria-label="${escape(spec.title)}" data-experiment="${kind}" font-family="Noto Sans, sans-serif" font-size="18">`, `<title>${escape(spec.title)}</title>`, `<desc>${escape(spec.yTitle)}. Bars start at zero.</desc>`, '<style>.exp-unit,.exp-tick,.exp-category,.exp-axis-title{font:18px "Noto Sans",sans-serif}.exp-value{font:600 17px "Noto Sans",sans-serif}</style>'];
    if(spec.highlight) {
      const x=left+spec.highlight[0]*groupWidth;
      out.push(`<rect class="exp-highlight" x="${x}" y="${top}" width="${(spec.highlight[1]-spec.highlight[0]+1)*groupWidth}" height="${height}" fill="#eaf5f1"/>`);
    }
    out.push(`<text class="exp-unit" x="${left}" y="22" fill="#42566a">${escape(spec.yTitle)}</text>`);
    spec.yTicks.forEach(value=>{
      const yy=y(value);
      out.push(`<line class="exp-gridline" x1="${left}" x2="${W-right}" y1="${yy}" y2="${yy}" stroke="#e0e7ec" stroke-width="1"${value?' stroke-dasharray="3 4"':''}/><text class="exp-tick" x="${left-10}" y="${yy+6}" text-anchor="end" fill="#607487">${value}</text>`);
    });
    out.push(`<line x1="${left}" x2="${left}" y1="${top}" y2="${bottom}" stroke="#9cabb6" stroke-width="1"/><line x1="${left}" x2="${W-right}" y1="${bottom}" y2="${bottom}" stroke="#9cabb6" stroke-width="1"/>`);
    spec.categories.forEach((category,i)=>{
      const center=left+(i+.5)*groupWidth;
      if(!spec.ticks || spec.ticks.includes(i)) {
        const lines=spec.shortLabels?.[i] || [category];
        lines.forEach((line,j)=>out.push(`<text class="exp-category" x="${center}" y="${bottom+26+j*21}" text-anchor="middle" fill="#42566a">${escape(line)}</text>`));
      }
      spec.series.forEach((series,j)=>{
        const value=series.values[i];
        const x=center-groupSpan/2+j*(barWidth+gap), yy=y(value);
        const fill=color(series.barColors?.[i] || series.color);
        const prefix='';
        const label=spec.kind==='probe' ? value.toFixed(3) : number(value);
        const seriesName=spec.kind==='components' ? category : series.name;
        const categoryName=spec.kind==='components' ? spec.xTitle : category;
        const tooltip=`${seriesName} · ${categoryName}: ${prefix}${label}${spec.unit}`;
        out.push(`<rect class="exp-bar" x="${x}" y="${yy}" width="${barWidth}" height="${bottom-yy}" rx="1.5" fill="${fill}" stroke="#122b40" stroke-opacity=".12" stroke-width=".7" tabindex="0" role="img" aria-label="${escape(tooltip)}" data-tooltip="${escape(tooltip)}" data-value="${value}" data-series="${escape(seriesName)}" data-category="${escape(categoryName)}"><title>${escape(tooltip)}</title></rect>`);
        if(series.lower) {
          const xx=x+barWidth/2, low=y(series.lower[i]), high=y(series.upper[i]);
          const cap=Math.min(5,barWidth*.36);
          out.push(`<path class="exp-whisker" d="M${xx},${high}V${low}M${xx-cap},${high}H${xx+cap}M${xx-cap},${low}H${xx+cap}" fill="none" stroke="#42566a" stroke-opacity=".64" stroke-width="1" pointer-events="none"/>`);
        }
        if(spec.directLabels) out.push(`<text class="exp-value" font-size="17" font-weight="600" x="${x+barWidth/2}" y="${yy-9}" text-anchor="middle" fill="${fill==='#087b76'?'#087b76':'#42566a'}">${escape(prefix+label)}</text>`);
      });
    });
    if(spec.xTitle) out.push(`<text class="exp-axis-title" x="${left+width/2}" y="371" text-anchor="middle" fill="#42566a">${escape(spec.xTitle)}</text>`);
    out.push('</svg>');
    return out.join('');
  }

  function legend(kind, choice) {
    const spec=specification(kind,choice);
    const series=spec.legend || spec.series;
    let html=series.map(s=>`<span class="legend-entry"><span class="legend-swatch" style="--series-color:${color(s.color)}" aria-hidden="true"></span>${escape(s.name)}</span>`).join('');
    return html;
  }

  window.Bridge3DChartEngine={render,legend,specification};
  if(typeof document==='undefined') return;

  document.querySelectorAll('[data-chart-choice]').forEach(select=>{
    select.addEventListener('change',()=>{
      const kind=select.dataset.chartChoice;
      const figure=document.getElementById(`experiment-${kind}`);
      figure.querySelector('.chart-scroll').innerHTML=render(kind,select.value);
      figure.querySelector('.chart-legend').innerHTML=legend(kind,select.value);
      figure.querySelector('.chart-tooltip').hidden=true;
      figure.querySelector('.chart-state').textContent=kind==='probe'?`Denoising step ${data.probe.series[Number(select.value)].step}`:`${data.components.categories[Number(select.value)]} success rates`;
    });
  });
  document.querySelectorAll('.experiment-chart').forEach(figure=>{
    const tooltip=figure.querySelector('.chart-tooltip');
    function show(bar,event) {
      if(!bar) return;
      tooltip.textContent=bar.dataset.tooltip;
      tooltip.hidden=false;
      const outer=figure.getBoundingClientRect();
      const rect=bar.getBoundingClientRect();
      const x=(event?.clientX ?? (rect.left+rect.width/2))-outer.left;
      const y=(event?.clientY ?? rect.top)-outer.top;
      tooltip.style.left=Math.max(5,Math.min(x+10,outer.width-tooltip.offsetWidth-5))+'px';
      tooltip.style.top=Math.max(0,y-tooltip.offsetHeight-12)+'px';
    }
    figure.addEventListener('pointerover',event=>show(event.target.closest?.('.exp-bar'),event));
    figure.addEventListener('pointermove',event=>show(event.target.closest?.('.exp-bar'),event));
    figure.addEventListener('pointerout',event=>{if(event.target.closest?.('.exp-bar'))tooltip.hidden=true;});
    figure.addEventListener('focusin',event=>show(event.target.closest?.('.exp-bar')));
    figure.addEventListener('focusout',()=>{tooltip.hidden=true;});
    figure.addEventListener('keydown',event=>{if(event.key==='Escape')tooltip.hidden=true;});
  });
}());
