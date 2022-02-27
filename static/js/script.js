$(document).ready(function()
{
	$('form').on('submit', function(event)
  {
		$.ajax
    ({
      data : {
				response : $('#msg').val()
			},
			type : 'POST',
			url : '/process'
		})
		.done(function(data)
    {
      let botMsg = '<div class="box-bot"><div class="msg-bot">'+data.response+'</div></div>';
		  $('#chat').append(botMsg);
      // Clear the input input box
      document.getElementById('msg').value="";
		});

		event.preventDefault();

	});

});

// Method to write out the user input
function writeUserMsg()
{
  let userMsg = document.getElementById('msg').value;
  document.getElementById('chat').innerHTML+='<div class="box-user"><div class="msg-user">'+userMsg+'</div></div>';
}
