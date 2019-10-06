(function(){
      var feed = document.getElementById('webcamstream'),
          vendorURL = window.URL || window.webkitURL;

      navigator.getUserMedia = navigator.getUserMedia || navigator.webkitGetUserMedia || navigator.mozGetUserMedia || navigator.msGetUserMedia;

      navigator.getUserMedia({
        video:true,
        audio:false
      }, function(stream){
        feed.srcObject = stream;
        feed.play();
        capture(feed);
      }, function(error){
        //error
      });
})();

function capture(feed){
  var canvas = document.getElementById('canvas');
  var context =  canvas.getContext('2d');
  context.drawImage(feed, 0, 0, 240, 180);
  const dataURI = canvas.toDataURL();
 // console.log(dataURI.substring(22));
  imagedata = dataURI.substring(22);
  postdata = "{\"image\":\"" + imagedata + "\"}";
  $.post("image", postdata ,function(status){
    console.log(status);
  });
}
