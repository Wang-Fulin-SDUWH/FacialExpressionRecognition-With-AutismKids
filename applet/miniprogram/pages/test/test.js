// pages/test/test.js

const app = getApp()
var that
const db = wx.cloud.database();
Page({

  /**
   * 页面的初始数据
   */
  data: {
    num:0,
    result:1
  },

  /**
   * 生命周期函数--监听页面加载
   */
  onLoad: function (options) {

  },
  bindInput: function (e) {
    this.setData({
      num: Number(e.detail.value)
    })
    console.log('input', this.data.num)
  },

  sum: function () {
    that=this
    wx.request({
      url: 'http://49.7.206.10:8000/test',
      data:{
        num:that.data.num,
      },
      method: "post",
      success(res) {
        that.setData({
          result: res.data+1,
        }),
          db.collection('test').add({
            data:{
              sum: res.data
            }

          })
          .then(res => {

          })
          .catch(console.error)

      }

    })
    
  },
  upload: function () {
    that = this
    
    db.collection('test').add({
      data: {
        sum:that.data.result+1
      }
      })
      .then(res => {

      })
      .catch(console.error)


  },


  /**
   * 生命周期函数--监听页面初次渲染完成
   */
  onReady: function () {

  },

  /**
   * 生命周期函数--监听页面显示
   */
  onShow: function () {

  },

  /**
   * 生命周期函数--监听页面隐藏
   */
  onHide: function () {

  },

  /**
   * 生命周期函数--监听页面卸载
   */
  onUnload: function () {

  },

  /**
   * 页面相关事件处理函数--监听用户下拉动作
   */
  onPullDownRefresh: function () {

  },

  /**
   * 页面上拉触底事件的处理函数
   */
  onReachBottom: function () {

  },

  /**
   * 用户点击右上角分享
   */
  onShareAppMessage: function () {

  }
})