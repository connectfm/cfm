package ui.playback;

import android.content.SharedPreferences;
import android.graphics.drawable.BitmapDrawable;
import android.graphics.drawable.Drawable;
import android.os.Build;
import android.os.Bundle;
import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.annotation.RequiresApi;
import androidx.fragment.app.Fragment;

import android.os.CountDownTimer;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.ProgressBar;
import android.widget.SeekBar;
import android.widget.TextView;
import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.annotation.RequiresApi;
import androidx.fragment.app.Fragment;
import com.example.cfm.R;
import com.google.gson.Gson;
import com.squareup.picasso.Picasso;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;
import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;
import spotifyFramework.PlaybackService;
import spotifyFramework.Song;
import spotifyFramework.SongService;
import spotifyFramework.VolleyCallBack;

public class SongFragment extends Fragment {

    //Views for initializing the fragment
    public static ImageView albumArt;
    public static TextView artists;
    public static TextView albumName;
    public static TextView songName;

    //Buttons initialized in the fragment
    public Button playButton;
    public Button nextButton;
    public Button prevButton;
    public SeekBar seekBar;
    public CountDownTimer count;


    //Playback fields
    public ArrayList<Song> songQueue = new ArrayList<Song>();
    private boolean isPaused = false;
    private Long songLeft;
    private int songPos = 0;
    Runnable runnable;



    //Storage for interactions with Spotify/Amplify
    private PlaybackService playbackService;
    private SongService songService;


    @RequiresApi(api = Build.VERSION_CODES.N)
    @Override
    public View onCreateView(LayoutInflater inflater, ViewGroup container,
                             Bundle savedInstanceState) {
        // Inflate the layout for this fragment
        View root = inflater.inflate(R.layout.fragment_song, container, false);
        initializeObjects(root);
        initializeSongInfo(() -> {
            setSongLayout(root,songQueue.get(0));
            firstPlay();

        });
        return root;
    }


    @Override
    public void onActivityCreated(@Nullable Bundle savedInstanceState) {
        super.onActivityCreated(savedInstanceState);
    }

    private void populateSongInfo(Song s, JSONArray artists, JSONArray albumImages) throws JSONException {
        for(int i = 0; i < artists.length(); i++) {
            s.setArtist(artists.getString(i));
        }

        for(int j = 0; j < albumImages.length(); j++) {
            s.setImage(albumImages.getString(j));
        }
    }

    private void setSongLayout(View view, Song song) {
        artists = view.findViewById(R.id.song_artist);
        albumName = view.findViewById(R.id.song_album);
        songName = view.findViewById(R.id.song_title);

        Picasso.get().load(song.getImages().get(0)).into(albumArt);
        artists.setText(song.artistsToString(Integer.MAX_VALUE));
        albumName.setText(song.getAlbum());
        songName.setText(song.getName());

    }

    @RequiresApi(api = Build.VERSION_CODES.N)
    private void initializeSongInfo(VolleyCallBack callBack) {
        songService = new SongService(getActivity());

        songService.getRecentlyPlayed(() -> {
            songQueue = songService.getPlaylist();
            callBack.onSuccess();
        });
    }

    private void firstPlay() {
        playbackService.findDevice(() -> {

            playbackService.play(songQueue);
        });

    }


    private void playSong(Long duration) {


        playbackService.findDevice(() -> {
            playbackService.play();
        });
    }

    private void pauseSong() {
        playbackService.findDevice(() -> {
            playbackService.pause();
        });
    }

    private void skipSong() {
        playbackService.findDevice(() -> {
            playbackService.next();
        });
    }

    private void playLast() {
        playbackService.findDevice(() -> {
            playbackService.prev();
        });
    }

    private void initializeObjects(View v) {
        albumArt = v.findViewById(R.id.album_art);
        seekBar = v.findViewById(R.id.progressBar);
        Long duration = songQueue.get(songPos).getDuration()/1000;
        seekBar.setMax(duration.intValue());


        playButton = v.findViewById(R.id.play);
        playButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if(isPaused) {
                    isPaused = false;
                    playSong(songLeft);
                    playButton.setText("Pause");
                }
                else {
                    pauseSong();
                    isPaused = true;
                    playButton.setText("Play");
                }
            }
        });
        nextButton = v.findViewById(R.id.next);
        nextButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                skipSong();
                songPos ++;
                setSongLayout(v, songQueue.get(songPos));
            }
        });
        prevButton = v.findViewById(R.id.prev);
        prevButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                playLast();
            }
        });

        playbackService = new PlaybackService(getActivity());
    }

    protected void initializeSeekBar(Song currentSong) {
        seekBar.setMax((currentSong.getDuration().intValue()) / 1000);
       runnable = new Runnable() {
           @Override
           public void run() {
               playbackService.findDevice(() -> {
                   if(playbackService.getDeviceId() != null) {
                   }
               });
           }
       };
    }
}