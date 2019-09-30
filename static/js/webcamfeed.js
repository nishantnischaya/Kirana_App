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
      }, function(error){
        //error
      });
})();
