package com.example.cfm.ui.home;

import android.content.Context;
import android.os.Bundle;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.fragment.app.Fragment;
import androidx.lifecycle.Observer;
import androidx.lifecycle.ViewModelProvider;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;

import com.example.cfm.R;
import com.example.spotify_framework.Playlist;
import com.example.spotify_framework.Song;
import com.example.spotify_framework.SongService;
import com.example.spotify_framework.VolleyCallBack;

import java.util.ArrayList;
import java.util.List;

import static java.lang.Thread.sleep;

public class HomeFragment extends Fragment {
    public ArrayList<Song> songs =  new ArrayList<Song>();
    public SongService songService;
    private RecyclerView recyclerView;
    private SongsAdapter adapter;

    private HomeViewModel homeViewModel;

    public void onCreate() {
    }

    public View onCreateView(@NonNull LayoutInflater inflater,
                             ViewGroup container, Bundle savedInstanceState) {
        homeViewModel = new ViewModelProvider(this).get(HomeViewModel.class);
        View root = inflater.inflate(R.layout.fragment_home, container, false);
        recyclerView = (RecyclerView) root.findViewById(R.id.listening_history);
        waitForSongInfo(root.getContext());
        adapter = new SongsAdapter(getActivity(), songs);
        recyclerView.setAdapter(adapter);
        recyclerView.setLayoutManager(new LinearLayoutManager(getActivity()));
        return root;
    }

    private void waitForSongInfo(Context context) {
        songService = new SongService(context);
        songService.getRecentlyPlayed(() -> {
            songs = songService.getPlaylist();
        });

        try {
            if(songService.getPlaylist().size() == 0) {
                sleep(1000);
            }
        } catch (InterruptedException e) {

        } finally {
            for(Song s: songs) {
                System.out.println(s.getName());
            }
        }
    }

}