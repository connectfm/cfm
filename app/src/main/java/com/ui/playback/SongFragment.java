package com.ui.playback;

import android.content.SharedPreferences;
import android.graphics.drawable.Drawable;
import android.os.Build;
import android.os.Bundle;

import androidx.annotation.Nullable;
import androidx.annotation.RequiresApi;
import androidx.fragment.app.Fragment;

import android.os.CountDownTimer;
import android.os.Handler;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.ImageButton;
import android.widget.ImageView;
import android.widget.SeekBar;
import android.widget.TextView;

import com.datastoreInteractions.AmplifyService;
import com.example.cfm.R;
import com.squareup.picasso.Picasso;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;

import org.json.JSONArray;
import org.json.JSONException;

import com.spotifyFramework.PlaybackService;
import com.spotifyFramework.Song;
import com.spotifyFramework.SongService;
import com.spotifyFramework.VolleyCallBack;

public class SongFragment extends Fragment {

    //Views for initializing the fragment
    public static ImageView albumArt;
    public static TextView artists;
    public static TextView albumName;
    public static TextView songName;

    //Buttons initialized in the fragment
    public ImageButton playButton;
    public ImageButton nextButton;
    public ImageButton prevButton;
    public ImageButton likeButton;
    public ImageButton dislikeButton;
    public SeekBar seekBar;
    public CountDownTimer count;


    //Playback fields
    public ArrayList<Song> songQueue = new ArrayList<Song>();
    private boolean isPaused = false;
    private int songPos = 0;
    private Runnable runnable;
    private Handler handler;
    private int timeAtSeek;
    private Song currentSong;
    private ArrayList<String> ids = new ArrayList<String>();
    private boolean buttonPressed = false;




    //Storage for interactions with Spotify/Amplify
    private PlaybackService playbackService;
    private SongService songService;

    private SharedPreferences preferences;



    @RequiresApi(api = Build.VERSION_CODES.N)
    @Override
    public View onCreateView(LayoutInflater inflater, ViewGroup container,
                             Bundle savedInstanceState) {
        // Inflate the layout for this fragment
        View root = inflater.inflate(R.layout.fragment_song, container, false);
        songService = new SongService(root.getContext());
        preferences = getActivity().getSharedPreferences("SPOTIFY",0);

        for(int i = 1; i <= 3 ;i++) {
            System.out.println(preferences.getString("id_"+i, ""));
            ids.add(preferences.getString("id_"+i, ""));
        }

        songService.populateMultipleSongs(ids, () -> {
            songQueue = songService.getPlaylist();
            initializeObjects(root);
            setSongLayout(root, songQueue.get(songPos));
            playbackService.play(songQueue.get(0),0);
        });

        return root;
    }


    @Override
    public void onActivityCreated(@Nullable Bundle savedInstanceState) {
        super.onActivityCreated(savedInstanceState);
    }

    private void setSongLayout(View view, Song song) {
        currentSong = song;
        artists = view.findViewById(R.id.song_artist);
        albumName = view.findViewById(R.id.song_album);
        songName = view.findViewById(R.id.song_title);
        albumArt = view.findViewById(R.id.album_art);

        Picasso.get().load(song.getImages().get(0)).into(albumArt);
        artists.setText(song.artistsToString(Integer.MAX_VALUE));
        albumName.setText(song.getAlbum());
        songName.setText(song.getName());
    }

    private void playSong() {

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
            playbackService.play(songQueue.get(songPos),0);

        });
    }

    private void playLast() {
        if(songPos >= 1){
            songPos--;
        }
        seekBar.setMax(songQueue.get(songPos).getDuration().intValue());
        playbackService.findDevice(() -> {
            playbackService.play(songQueue.get(songPos), 0);
        });

    }

    private void initializeObjects(View v) {
        albumArt = v.findViewById(R.id.album_art);
        seekBar = v.findViewById(R.id.progressBar);
        handler = new Handler();
        initializeSeekBar(songQueue.get(songPos));

        playButton = v.findViewById(R.id.play);
        playButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if(isPaused) {
                    isPaused = false;
                    Drawable res = getContext().getDrawable(R.drawable.pause);
                    playButton.setBackground(res);
                    playSong();
                }
                else {
                    isPaused = true;
                    Drawable res = getContext().getDrawable(R.drawable.play);
                    playButton.setBackground(res);
                    pauseSong();
                }
            }
        });
        nextButton = v.findViewById(R.id.next);
        nextButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                buttonPressed = true;
                skipSong();
                songPos ++;
                seekBar.setMax(songQueue.get(songPos).getDuration().intValue());
                setSongLayout(v, songQueue.get(songPos));
            }
        });
        prevButton = v.findViewById(R.id.prev);
        prevButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                if(seekBar.getProgress() >= seekBar.getMax()/2) {
                    buttonPressed = true;
                    playbackService.play(songQueue.get(songPos),0);
                }
                else {
                    playLast();
                }
            }
        });
        likeButton = v.findViewById(R.id.like);
        likeButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Drawable res;
                if(currentSong.getStatus() == 3) {
                    currentSong.setStatus(2);
                    res = getContext().getDrawable(R.drawable.thumbs_up);
                }
                else{
                    if(currentSong.getStatus()==1) {
                        Drawable res2 = getContext().getDrawable(R.drawable.thumbs_up);
                        dislikeButton.setBackground(res2);
                    }
                    currentSong.setStatus(3);
                    res = getContext().getDrawable(R.drawable.thumbs_up_pressed);
                }
                likeButton.setBackground(res);

            }
        });
        dislikeButton = v.findViewById(R.id.dislike);
        dislikeButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Drawable res;
                if(currentSong.getStatus() == 1) {
                    currentSong.setStatus(2);
                    res = getContext().getDrawable(R.drawable.thumbs_up);
                }

                else {
                    if(currentSong.getStatus() == 3) {
                        Drawable res2 = getContext().getDrawable(R.drawable.thumbs_up);
                        likeButton.setBackground(res2);
                    }
                    currentSong.setStatus(1);
                    res = getContext().getDrawable(R.drawable.thumbs_up_pressed);
                }
                dislikeButton.setBackground(res);
            }
        });
        playbackService = new PlaybackService(getActivity());

    }



    private void initializeSeekBar(Song currentSong) {
        seekBar.setMax((currentSong.getDuration().intValue()));

        seekBar.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
            @Override
            public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
                timeAtSeek = progress;

                if(progress == songQueue.get(songPos).getDuration()) {
                    skipSong();
                    songPos ++;
                    seekBar.setMax(songQueue.get(songPos).getDuration().intValue());
                    setSongLayout(getView(), songQueue.get(songPos));
                }
            }

            @Override
            public void onStartTrackingTouch(SeekBar seekBar) {

            }

            @Override
            public void onStopTrackingTouch(SeekBar seekBar) {
                timeAtSeek = seekBar.getProgress();
                playbackService.findDevice(() -> {
                    playbackService.play(songQueue.get(songPos),seekBar.getProgress());
                    if(isPaused)
                        playbackService.pause();
                });
            }
        });
        runnable = new Runnable() {
           @Override
           public void run() {

               playbackService.currentlyPlaying(() -> {
                   int currentPos = playbackService.getProgress();
                   seekBar.setProgress(currentPos);
               });
               handler.postDelayed(runnable,1000);
           }
        };
       handler.postDelayed(runnable,1000);
    }

    private ArrayList<Song> removeDupes(ArrayList<Song> playlist) {
        ArrayList<String> idList = new ArrayList<>();
        ArrayList<Song> res = new ArrayList<>();

        for(Song s: playlist) {
            if(!idList.contains(s.getUri())){
                res.add(s);
                idList.add(s.getUri());
            }
        }
        return res;
    }

}