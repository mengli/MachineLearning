package mengli.android.pcmaudiorecorder;

import java.io.FileOutputStream;
import java.io.IOException;
import java.util.concurrent.atomic.AtomicInteger;

import android.app.Activity;
import android.media.AudioFormat;
import android.media.AudioRecord;
import android.media.MediaRecorder;
import android.os.Bundle;
import android.view.Menu;
import android.view.MenuItem;
import android.view.View;
import android.view.View.OnClickListener;
import android.widget.Button;
import android.widget.TextView;

public class MainActivity extends Activity {

	private static final int RECORDING = 0;
	private static final int STOPPED = 1;

	private static final int RECORDER_SAMPLERATE = 8000;
	private static final int RECORDER_CHANNELS = AudioFormat.CHANNEL_IN_MONO;
	private static final int RECORDER_AUDIO_ENCODING = AudioFormat.ENCODING_PCM_16BIT;
	private AudioRecord recorder = null;
	private Thread recordingThread = null;

	private Button mStartButton;
	private Button mStopButton;

	private TextView mStatusLabel;
	private TextView mOutputLabel;

	private String mFilePath;

	private AtomicInteger status;
	private int mMinBufferSize;

	private void startRecording() {
	    recorder = new AudioRecord(MediaRecorder.AudioSource.MIC,
	            RECORDER_SAMPLERATE, RECORDER_CHANNELS,
	            RECORDER_AUDIO_ENCODING,  mMinBufferSize);

	    recorder.startRecording();
	    recordingThread = new Thread(new Runnable() {
	        public void run() {
	            writeAudioDataToFile();
	        }
	    }, "AudioRecorder Thread");
	    recordingThread.start();
	}

	private void writeAudioDataToFile() {
	    // Write the output audio in byte
	    byte[] buffer = new byte[mMinBufferSize];
	    FileOutputStream os = null;
	    try {
		    os = new FileOutputStream(mFilePath);
		    while (status.get() == RECORDING) {
		        // gets the voice output from microphone to byte format
		    	synchronized (recorder) {
		    		int count = recorder.read(buffer, 0, mMinBufferSize);
			        if (count > 0) {
			            os.write(buffer, 0, count);
			        }
				}
		    }
	    } catch (IOException e) {
	        e.printStackTrace();
	    } finally {
	    	if (os != null) {
	    		try {
					os.close();
				} catch (IOException e) {
					e.printStackTrace();
				}
	    	}
	    }
	}

	private void stopRecording() {
	    // stops the recording activity
	    if (null != recorder) {
	    	synchronized (recorder) {
	    		recorder.stop();
	    		recorder.release();
	    		recorder = null;
	    	}
	        recordingThread = null;
	    }
	}

	private OnClickListener mButtonClickListener = new OnClickListener() {
	    public void onClick(View v) {
	        switch (v.getId()) {
		        case R.id.btnStart: {
		            mStartButton.setEnabled(false);
		            mStopButton.setEnabled(true);
		            if (status.get() == STOPPED) {
		                startRecording();
		                status.set(RECORDING);
		                mStatusLabel.setText(R.string.recording);
		                mOutputLabel.setVisibility(View.INVISIBLE);
		            }
		            break;
		        }
		        case R.id.btnStop: {
		        	mStartButton.setEnabled(true);
		            mStopButton.setEnabled(false);
		            if (status.get() == RECORDING) {
		            	status.set(STOPPED);
		                stopRecording();
		                mStatusLabel.setText(R.string.stopped);
		                mOutputLabel.setVisibility(View.VISIBLE);
		                mOutputLabel.setText(mFilePath.toString());
		            }
		            break;
		        }
	        }
	    }
	};

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        mStartButton = (Button) findViewById(R.id.btnStart);
	    mStopButton = (Button) findViewById(R.id.btnStop);
	    mStatusLabel = (TextView) findViewById(R.id.status_lable);
	    mOutputLabel = (TextView) findViewById(R.id.output_lable);
	    mStartButton.setEnabled(true);
        mStopButton.setEnabled(false);
        mStartButton.setOnClickListener(mButtonClickListener);
        mStopButton.setOnClickListener(mButtonClickListener);
        status = new AtomicInteger();
        status.set(STOPPED);

	    mFilePath = android.os.Environment.getExternalStorageDirectory() + "/pcmrecorder/output.pcm";

	    mMinBufferSize = AudioRecord.getMinBufferSize(RECORDER_SAMPLERATE,
				RECORDER_CHANNELS, RECORDER_AUDIO_ENCODING);
    }


    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        // Inflate the menu; this adds items to the action bar if it is present.
        getMenuInflater().inflate(R.menu.main, menu);
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        // Handle action bar item clicks here. The action bar will
        // automatically handle clicks on the Home/Up button, so long
        // as you specify a parent activity in AndroidManifest.xml.
        int id = item.getItemId();
        if (id == R.id.action_settings) {
            return true;
        }
        return super.onOptionsItemSelected(item);
    }
}
