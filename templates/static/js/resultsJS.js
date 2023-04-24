window.onload = function () {
    var chart = new CanvasJS.Chart("chartContainer",
    {
      title:{
        text: "Olympic Medals of all Times (till 2012 Olympics)"
      },
      data: [
      {
        type: "bar",
        dataPoints: [
        { y: 198, label: "Italy"},
        { y: 201, label: "China"},
        { y: 202, label: "France"},
        { y: 236, label: "Great Britain"},
        { y: 395, label: "Soviet Union"},
        { y: 957, label: "USA"}
        ]
      },
      {
        type: "bar",
        dataPoints: [
        { y: 166, label: "Italy"},
        { y: 144, label: "China"},
        { y: 223, label: "France"},
        { y: 272, label: "Great Britain"},
        { y: 319, label: "Soviet Union"},
        { y: 759, label: "USA"}
        ]
      },
      {
        type: "bar",
        dataPoints: [
        { y: 185, label: "Italy"},
        { y: 128, label: "China"},
        { y: 246, label: "France"},
        { y: 272, label: "Great Britain"},
        { y: 296, label: "Soviet Union"},
        { y: 666, label: "USA"}
        ]
      }
      ]
    });

chart.render();