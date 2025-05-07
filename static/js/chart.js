function drawChart(data) {
  const hours = Array.from(new Set(data.features.map(f => f.properties.hour_code))).sort((a, b) => a - b);
  const sumByHour = hours.map(h => ({
    hour: h,
    total: d3.sum(data.features.filter(f => f.properties.hour_code == h), f => f.properties.activity_index)
  }));

  const svg = d3.select('#chart')
    .append('svg')
    .attr('width', 300)
    .attr('height', 150);

  const x = d3.scaleBand().domain(hours).range([30, 270]).padding(0.1);
  const y = d3.scaleLinear().domain([0, d3.max(sumByHour, d => d.total)]).range([120, 10]);

  svg.selectAll('rect')
    .data(sumByHour)
    .enter()
    .append('rect')
    .attr('x', d => x(d.hour))
    .attr('y', d => y(d.total))
    .attr('width', x.bandwidth())
    .attr('height', d => 120 - y(d.total))
    .attr('fill', 'steelblue');

  svg.append('g')
    .attr('transform', 'translate(0, 120)')
    .call(d3.axisBottom(x));

  svg.append('g')
    .attr('transform', 'translate(30, 0)')
    .call(d3.axisLeft(y));
}
