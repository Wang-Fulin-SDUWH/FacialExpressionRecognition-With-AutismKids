//index.js
const app = getApp()
const db = wx.cloud.database();
import * as echarts from '../../ec-canvas/echarts';
const moods = ['恐惧', '高兴', '中性', '伤心', '惊讶', '厌烦'];
let chart1 = null;
let chart2 = null;
let that;


function initChart1(canvas, width, height) {

  chart1 = echarts.init(canvas, null, {
    width: width,
    height: height
  });

  canvas.setChart(chart1);

  var option = {
    title: {
      show: true,
      text: '情绪时序数据'
    },
    legend: {
      type: 'plain',
      show: true,
      orient: 'vertical',
      right: '10%',
      top: '20%'
    },
    backgroundColor: "#ffffff",
    color: ["#37A2DA", "#FF9F7F"],
    xAxis: {
      name: '时间',
      type: 'time',
      // data:[],
      // show:false,
    },
    yAxis: {
      type: 'category',
      splitLine: {
        lineStyle: {
          type: 'solid',
        }
      }
    },
    series: [{
        name: '负面情绪',
        type: 'scatter',
        emphasis: {
          label: {
            show: true,
            position: 'left',
            color: 'red'
          }
        },
        color: 'red',
        // symbolSize: 20
      },

      {
        name: '情绪',
        type: 'line',
        smooth: true,
      }

    ]
  }
  chart1.setOption(option);
}

function initChart2(canvas, width, height) {

  chart2 = echarts.init(canvas, null, {
    width: width,
    height: height
  });

  canvas.setChart(chart2);

  var option = {
    tooltip: {
      trigger: 'item',
      formatter: '{a} <br/>{b}: {c} ({d}%)'
    },
    legend: {
      orient: 'vertical',
      left: 10,
      data: moods
    },
    series: [{
        name: '',
        type: 'pie',
        selectedMode: 'single',
        radius: [0, '30%'],

        label: {
          position: 'inner'
        },
        labelLine: {
          show: false
        },
        data: [{
            value: 3,
            name: '负面',
          },
          {
            value: 3,
            name: '积极'
          }
        ]
      },
      {
        name: '情绪',
        type: 'pie',
        radius: ['40%', '55%'],
        label: {
          formatter: '{per|{d}%}',
          borderColor: '#aaa',
          borderWidth: 1,
          borderRadius: 4,
          rich: {
            per: {
              color: '#334455',
              padding: [2, 4],
              borderRadius: 2
            }
          }
        },
        data: [{
            value: 1,
            name: '恐惧'
          },
          {
            value: 1,
            name: '高兴'
          },
          {
            value: 1,
            name: '中性'
          },
          {
            value: 1,
            name: '伤心'
          },
          {
            value: 1,
            name: '惊讶'
          },
          {
            value: 1,
            name: '厌烦'
          }
        ]
      }
    ]
  }
  chart2.setOption(option);
}


Page({
  data: {
    idInput: 0,
    ec1: {
      onInit: initChart1
    },
    ec2: {
      onInit: initChart2
    },

    period: ['三十秒', '一分钟', '...近一天'],
    period_index: 0,
  },
  pickPeriod: function(e) {
    console.log('period携带值为', e.detail.value)
    this.setData({
      period_index: Number(e.detail.value)
    })
  },

  idInput: function(e) {
    this.setData({
      idInput: Number(e.detail.value)
    })
    console.log('idInput', this.data.idInput)
  },

  query: function() {
    let mData = [];
    let negative = [];
    let mood = [];
    let time = [];
    var count = [0, 0, 0, 0, 0, 0];
    that = this;
    var splice;



    db.collection('test').where({
        id: that.data.idInput
      })
      .get()
      .then(res => {
        var len = res.data["0"].mood.length;
        switch (this.data.period_index) {
          case 0:
            splice = 15;
            break;
          case 1:
            splice = 30;
            break;
          case 2:
            splice = len;
            break;
        }
        mData = res.data["0"].mood
          .splice(-splice, splice);
        time = res.data["0"].timeStamp
          .splice(-splice, splice);
        let i;
        for (i = 0; i < time.length; i++) {
          time[i] *= 1000;
        }
        for (i = 0; i < mData.length; i++) {

          // console.log(moods[5])
          switch (mData[i]) {
            case 0:
              mood.push([time[i], moods[0]]);
              negative.push([time[i], moods[0]]);
              count[0] += 1;
              break;
            case 1:
              mood.push([time[i], moods[1]]);
              count[1] += 1;
              break;
            case 2:
              mood.push([time[i], moods[2]]);
              count[2] += 1;
              break;
            case 3:
              mood.push([time[i], moods[3]]);
              negative.push([time[i], moods[3]]);
              count[3] += 1;
              break;
            case 4:
              mood.push([time[i], moods[4]]);
              count[4] += 1;
              break;
            case 5:
              mood.push([time[i], moods[5]]);
              negative.push([time[i], moods[5]]);
              count[5] += 1;
              break;
          }
        }

        chart1.setOption({
          series: [{
              name: '负面情绪',
              data: negative,
              // symbolSize: 20
            },

            {
              name: '情绪',
              data: mood
            }

          ]
        });
        chart2.setOption({
          series: [{
              name: '',
              data: [{
                  value: count[0] + count[3] + count[5],
                  name: '负面',
                },
                {
                  value: count[1] + count[2] + count[4],
                  name: '积极'
                }
              ],
            },

            {
              name: '情绪',
              data: [{
                  value: count[0],
                  name: '恐惧'
                },
                {
                  value: count[1],
                  name: '高兴'
                },
                {
                  value: count[2],
                  name: '中性'
                },
                {
                  value: count[3],
                  name: '伤心'
                },
                {
                  value: count[4],
                  name: '惊讶'
                },
                {
                  value: count[5],
                  name: '厌烦'
                }
              ]
            }

          ]
        });
      })
      .catch(res => {
        console.log(res);
        wx.showModal({
          title: 'ID不存在',
        })
      })

  },

})